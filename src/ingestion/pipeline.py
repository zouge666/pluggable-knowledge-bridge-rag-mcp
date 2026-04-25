"""
Ingestion Pipeline - 数据摄取流水线。

串行执行：integrity → load → split → transform → encode → store
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, ChunkRecord, Document
from src.ingestion.chunking import DocumentChunker
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.transform import ChunkRefiner, ImageCaptioner
from src.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.vector_store import BaseVectorStore


@dataclass
class PipelineResult:
    """Pipeline 执行结果。"""

    success: bool
    doc_hash: str
    file_path: str
    collection: Optional[str] = None
    chunks_count: int = 0
    images_count: int = 0
    elapsed_ms: float = 0.0
    error: Optional[str] = None
    skipped: bool = False
    stages: List[dict] = field(default_factory=list)


class IngestionPipeline:
    """
    数据摄取流水线。

    编排完整的数据摄取流程：
    1. Integrity Check - 检查文件是否需要处理
    2. Load - 加载文档
    3. Split - 切分为 Chunks
    4. Transform - 增强/精炼 Chunks
    5. Encode - 向量化
    6. Store - 存储到索引

    特性：
    - 幂等性：相同文件不重复处理
    - 可观测：每个阶段记录耗时
    - 可恢复：失败时记录状态
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        integrity_checker: Optional[FileIntegrityChecker] = None,
        loader: Optional[PdfLoader] = None,
        chunker: Optional[DocumentChunker] = None,
        chunk_refiner: Optional[ChunkRefiner] = None,
        image_captioner: Optional[ImageCaptioner] = None,
        batch_processor: Optional[BatchProcessor] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_upserter: Optional[VectorUpserter] = None,
        image_storage: Optional[ImageStorage] = None,
        vector_store: Optional[BaseVectorStore] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ):
        """
        初始化 IngestionPipeline。

        Args:
            settings: 应用配置（用于创建默认组件）。
            integrity_checker: 文件完整性检查器。
            loader: 文档加载器。
            chunker: 文档切分器。
            chunk_refiner: Chunk 精炼器。
            image_captioner: 图片描述生成器。
            batch_processor: 批处理器。
            bm25_indexer: BM25 索引器。
            vector_upserter: 向量写入器。
            image_storage: 图片存储。
            vector_store: 向量存储。
            on_progress: 进度回调函数 (stage_name, current, total)。
        """
        self._settings = settings or Settings()

        # 初始化组件（支持依赖注入）
        self._integrity_checker = integrity_checker or SQLiteIntegrityChecker()
        self._loader = loader or PdfLoader()
        self._chunker = chunker or DocumentChunker(self._settings)
        self._chunk_refiner = chunk_refiner
        self._image_captioner = image_captioner
        self._batch_processor = batch_processor
        self._bm25_indexer = bm25_indexer
        self._vector_upserter = vector_upserter
        self._image_storage = image_storage
        self._vector_store = vector_store
        self._on_progress = on_progress

    def ingest(
        self,
        file_path: Union[str, Path],
        collection: Optional[str] = None,
        force: bool = False,
        trace: Optional[TraceContext] = None,
    ) -> PipelineResult:
        """
        执行数据摄取。

        Args:
            file_path: 文件路径。
            collection: 集合名称。
            force: 强制重新处理（忽略完整性检查）。
            trace: 追踪上下文。

        Returns:
            PipelineResult: 执行结果。
        """
        start_time = time.time()
        file_path = Path(file_path)
        stages = []

        # 1. Integrity Check
        stage_start = time.time()
        doc_hash = self._integrity_checker.compute_sha256(str(file_path))

        if not force and self._integrity_checker.should_skip(doc_hash):
            elapsed_ms = (time.time() - start_time) * 1000
            return PipelineResult(
                success=True,
                doc_hash=doc_hash,
                file_path=str(file_path),
                collection=collection,
                skipped=True,
                elapsed_ms=elapsed_ms,
                stages=[{"stage": "integrity_check", "action": "skipped"}],
            )

        stages.append({
            "stage": "integrity_check",
            "elapsed_ms": (time.time() - stage_start) * 1000,
        })
        self._report_progress("integrity_check", 1, 6)

        try:
            # 2. Load
            stage_start = time.time()
            document = self._loader.load(file_path, collection, trace)
            stages.append({
                "stage": "load",
                "elapsed_ms": (time.time() - stage_start) * 1000,
                "details": {"doc_id": document.id},
            })
            self._report_progress("load", 2, 6)

            # 3. Split
            stage_start = time.time()
            chunks = self._chunker.split_document(document, trace)
            stages.append({
                "stage": "split",
                "elapsed_ms": (time.time() - stage_start) * 1000,
                "details": {"chunks_count": len(chunks)},
            })
            self._report_progress("split", 3, 6)

            # 4. Transform
            stage_start = time.time()
            chunks = self._transform_chunks(chunks, document, trace)
            stages.append({
                "stage": "transform",
                "elapsed_ms": (time.time() - stage_start) * 1000,
                "details": {"chunks_count": len(chunks)},
            })
            self._report_progress("transform", 4, 6)

            # 5. Encode
            stage_start = time.time()
            records = self._encode_chunks(chunks, trace)
            stages.append({
                "stage": "encode",
                "elapsed_ms": (time.time() - stage_start) * 1000,
                "details": {"records_count": len(records)},
            })
            self._report_progress("encode", 5, 6)

            # 6. Store
            stage_start = time.time()
            self._store_records(records, collection, doc_hash, trace)
            stages.append({
                "stage": "store",
                "elapsed_ms": (time.time() - stage_start) * 1000,
            })
            self._report_progress("store", 6, 6)

            # 标记成功
            self._integrity_checker.mark_success(
                doc_hash,
                str(file_path),
                collection,
                metadata={"chunks_count": len(chunks)},
            )

            # 保存图片
            images_count = self._save_images(document, collection, doc_hash, trace)

            elapsed_ms = (time.time() - start_time) * 1000

            return PipelineResult(
                success=True,
                doc_hash=doc_hash,
                file_path=str(file_path),
                collection=collection,
                chunks_count=len(chunks),
                images_count=images_count,
                elapsed_ms=elapsed_ms,
                stages=stages,
            )

        except Exception as e:
            # 标记失败
            self._integrity_checker.mark_failed(doc_hash, str(e))

            elapsed_ms = (time.time() - start_time) * 1000
            return PipelineResult(
                success=False,
                doc_hash=doc_hash,
                file_path=str(file_path),
                collection=collection,
                elapsed_ms=elapsed_ms,
                error=str(e),
                stages=stages,
            )

    def _transform_chunks(
        self,
        chunks: List[Chunk],
        document: Document,
        trace: Optional[TraceContext] = None,
    ) -> List[Chunk]:
        """Transform 阶段：精炼和增强 Chunks。"""
        result = chunks

        # Chunk Refiner
        if self._chunk_refiner:
            result = self._chunk_refiner.refine(result, trace)

        # Image Captioner
        if self._image_captioner:
            images = document.get_images()
            if images:
                result = self._image_captioner.caption(result, images, trace)

        return result

    def _encode_chunks(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[ChunkRecord]:
        """Encode 阶段：向量化 Chunks。"""
        if self._batch_processor:
            return self._batch_processor.process(chunks, trace)

        # 如果没有 batch_processor，尝试创建默认的
        if self._settings:
            dense_encoder = DenseEncoder(settings=self._settings)
            sparse_encoder = SparseEncoder()
            processor = BatchProcessor(dense_encoder, sparse_encoder)
            return processor.process(chunks, trace)

        raise RuntimeError("No batch processor configured")

    def _store_records(
        self,
        records: List[ChunkRecord],
        collection: Optional[str],
        doc_hash: str,
        trace: Optional[TraceContext] = None,
    ) -> None:
        """Store 阶段：存储到索引。"""
        if not records:
            return

        # 存储到向量数据库
        if self._vector_upserter and self._vector_store:
            self._vector_upserter.upsert(records, collection, trace)

        # 构建 BM25 索引
        if self._bm25_indexer:
            self._bm25_indexer.index(records, collection, trace)

    def _save_images(
        self,
        document: Document,
        collection: Optional[str],
        doc_hash: str,
        trace: Optional[TraceContext] = None,
    ) -> int:
        """保存文档中的图片。"""
        if not self._image_storage:
            return 0

        images = document.get_images()
        if not images:
            return 0

        saved_count = 0
        for img in images:
            # ImageRef is just a reference, actual data stored separately
            # Check if image file exists at the path
            if img.path and os.path.exists(img.path):
                with open(img.path, "rb") as f:
                    image_data = f.read()
                self._image_storage.save(
                    image_id=img.id,
                    image_data=image_data,
                    collection=collection,
                    doc_hash=doc_hash,
                    page_num=img.page,
                    trace=trace,
                )
                saved_count += 1

        return saved_count

    def _report_progress(self, stage: str, current: int, total: int) -> None:
        """报告进度。"""
        if self._on_progress:
            self._on_progress(stage, current, total)


class FakeIngestionPipeline:
    """
    Fake Ingestion Pipeline 用于测试。

    不执行实际处理，返回预设结果。
    """

    def __init__(self, result: Optional[PipelineResult] = None):
        """初始化 FakeIngestionPipeline。"""
        self._result = result
        self.ingest_calls: List[dict] = []

    def ingest(
        self,
        file_path: Union[str, Path],
        collection: Optional[str] = None,
        force: bool = False,
        trace: Optional[TraceContext] = None,
    ) -> PipelineResult:
        """执行 fake 摄取。"""
        self.ingest_calls.append({
            "file_path": str(file_path),
            "collection": collection,
            "force": force,
        })

        if self._result:
            return self._result

        return PipelineResult(
            success=True,
            doc_hash="fake-hash",
            file_path=str(file_path),
            collection=collection,
            chunks_count=10,
            images_count=0,
            elapsed_ms=100.0,
        )
