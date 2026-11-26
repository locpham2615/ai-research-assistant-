"use client";

import React, { useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  FileUp,
  Globe,
  Loader2,
  X,
} from "lucide-react";
import { ingestArxiv, uploadPdfDocument } from "@/lib/api";
import type { IngestResponse } from "@/types/api";

interface AddPaperModalProps {
  defaultTopicId: string;
  onClose: () => void;
  onIngested: (ingestedTopicId: string) => Promise<void>;
}

type IngestTab = "arxiv" | "upload";

export function AddPaperModal({
  defaultTopicId,
  onClose,
  onIngested,
}: AddPaperModalProps) {
  const [activeTab, setActiveTab] = useState<IngestTab>("arxiv");
  const [topicId, setTopicId] = useState<string>(
    defaultTopicId || "llm_eval_v1"
  );
  const [arxivId, setArxivId] = useState<string>("");
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [force, setForce] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [result, setResult] = useState<IngestResponse | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setErrorMessage("");
    setResult(null);
    const chosen = e.target.files?.[0] ?? null;
    if (!chosen) {
      setPdfFile(null);
      return;
    }
    if (!chosen.name.toLowerCase().endsWith(".pdf")) {
      setPdfFile(null);
      setErrorMessage("Chỉ hỗ trợ tệp định dạng PDF (.pdf).");
      return;
    }
    if (chosen.size > 50 * 1024 * 1024) {
      setPdfFile(null);
      setErrorMessage("Dung lượng tệp PDF vượt quá giới hạn 50 MB.");
      return;
    }
    setPdfFile(chosen);
  };

  const handleSubmit = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    setErrorMessage("");
    setResult(null);

    const cleanedTopic = topicId.trim();
    if (!cleanedTopic) {
      setErrorMessage("Vui lòng nhập mã chủ đề (topic_id).");
      return;
    }

    if (activeTab === "arxiv") {
      const cleanedArxiv = arxivId
        .trim()
        .replace(/^https?:\/\/arxiv\.org\/(abs|pdf)\//i, "")
        .replace(/\.pdf$/i, "");
      if (cleanedArxiv.length < 4) {
        setErrorMessage(
          "Vui lòng nhập mã arXiv hợp lệ (ví dụ: 1706.03762 hoặc 2309.06180)."
        );
        return;
      }

      setIsSubmitting(true);
      try {
        const res = await ingestArxiv({
          arxiv_id: cleanedArxiv,
          topic_id: cleanedTopic,
          force,
        });
        setResult(res);
        await onIngested(res.topic_id);
      } catch (err) {
        setErrorMessage(
          err instanceof Error
            ? err.message
            : "Đã xảy ra lỗi khi nạp bài báo từ arXiv."
        );
      } finally {
        setIsSubmitting(false);
      }
      return;
    }

    if (!pdfFile) {
      setErrorMessage("Vui lòng chọn một tệp PDF (.pdf) từ máy tính.");
      return;
    }
    if (!pdfFile.name.toLowerCase().endsWith(".pdf")) {
      setErrorMessage("Chỉ hỗ trợ tệp định dạng PDF (.pdf).");
      return;
    }

    setIsSubmitting(true);
    try {
      const res = await uploadPdfDocument({
        file: pdfFile,
        topic_id: cleanedTopic,
        force,
      });
      setResult(res);
      await onIngested(res.topic_id);
    } catch (err) {
      setErrorMessage(
        err instanceof Error ? err.message : "Đã xảy ra lỗi khi tải lên tệp PDF."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/45 p-4 backdrop-blur-[1px]"
      onClick={() => {
        if (!isSubmitting) {
          onClose();
        }
      }}
    >
      <div
        className="w-full max-w-lg rounded-2xl border border-slate-200 bg-white shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Thêm bài báo khoa học vào kho tri thức
            </h2>
            <p className="mt-0.5 text-xs text-slate-500">
              Tự động bóc tách cấu trúc GROBID, chia đoạn và lập chỉ mục Qdrant +
              SQLite FTS5
            </p>
          </div>
          <button
            type="button"
            disabled={isSubmitting}
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 disabled:opacity-50"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Mode Tabs */}
        <div className="grid grid-cols-2 gap-2 bg-slate-50 p-2 border-b border-slate-200">
          <button
            type="button"
            disabled={isSubmitting}
            onClick={() => {
              setActiveTab("arxiv");
              setErrorMessage("");
              setResult(null);
            }}
            className={`flex items-center justify-center gap-2 rounded-xl py-2 text-xs font-semibold transition cursor-pointer ${
              activeTab === "arxiv"
                ? "bg-white text-blue-700 shadow-xs border border-slate-200/80"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Globe className="h-3.5 w-3.5" />
            <span>arXiv ID</span>
          </button>

          <button
            type="button"
            disabled={isSubmitting}
            onClick={() => {
              setActiveTab("upload");
              setErrorMessage("");
              setResult(null);
            }}
            className={`flex items-center justify-center gap-2 rounded-xl py-2 text-xs font-semibold transition cursor-pointer ${
              activeTab === "upload"
                ? "bg-white text-blue-700 shadow-xs border border-slate-200/80"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <FileUp className="h-3.5 w-3.5" />
            <span>Upload PDF</span>
          </button>
        </div>

        {/* Form */}
        <form onSubmit={(e) => void handleSubmit(e)} className="space-y-4 p-5">
          {activeTab === "arxiv" ? (
            <div>
              <label
                htmlFor="arxiv-id-input"
                className="block text-xs font-semibold text-slate-700"
              >
                Mã bài báo arXiv (arXiv ID hoặc URL)
              </label>
              <input
                id="arxiv-id-input"
                type="text"
                value={arxivId}
                onChange={(e) => setArxivId(e.target.value)}
                disabled={isSubmitting}
                placeholder="Ví dụ: 1706.03762 hoặc 2309.06180"
                className="mt-1.5 w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:bg-slate-50"
              />
            </div>
          ) : (
            <div>
              <label
                htmlFor="pdf-file-input"
                className="block text-xs font-semibold text-slate-700"
              >
                Chọn tệp bài báo PDF (.pdf, tối đa 50 MB)
              </label>
              <input
                id="pdf-file-input"
                type="file"
                accept=".pdf,application/pdf"
                disabled={isSubmitting}
                onChange={handleFileChange}
                className="mt-1.5 block w-full rounded-xl border border-slate-300 bg-slate-50 px-3 py-2 text-xs text-slate-700 file:mr-3 file:rounded-lg file:border-0 file:bg-blue-700 file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-white hover:file:bg-blue-800"
              />
              {pdfFile && (
                <p className="mt-1.5 text-[11px] text-slate-500">
                  Đã chọn: <strong>{pdfFile.name}</strong> (
                  {(pdfFile.size / (1024 * 1024)).toFixed(2)} MB)
                </p>
              )}
            </div>
          )}

          {/* Topic ID */}
          <div>
            <label
              htmlFor="ingest-topic-id"
              className="block text-xs font-semibold text-slate-700"
            >
              Chủ đề lưu trữ (`topic_id`)
            </label>
            <input
              id="ingest-topic-id"
              type="text"
              value={topicId}
              onChange={(e) => setTopicId(e.target.value)}
              disabled={isSubmitting}
              maxLength={80}
              placeholder="llm_eval_v1"
              className="mt-1.5 w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2 font-mono text-xs text-slate-900 focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:bg-slate-50"
            />
          </div>

          {/* Force Checkbox */}
          <label className="flex items-center gap-2.5 text-xs text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={force}
              onChange={(e) => setForce(e.target.checked)}
              disabled={isSubmitting}
              className="h-4 w-4 rounded border-slate-300 text-blue-700 focus:ring-blue-500"
            />
            <span>
              Nạp lại bắt buộc (`force`) kể cả khi bài báo đã tồn tại trong kho
            </span>
          </label>

          {/* Loading Notice */}
          {isSubmitting && (
            <div className="flex items-center gap-2.5 rounded-xl border border-blue-200 bg-blue-50 px-3.5 py-2.5 text-xs text-blue-800">
              <Loader2 className="h-4 w-4 shrink-0 animate-spin text-blue-700" />
              <span>
                Đang tải tài liệu, phân tích cấu trúc bài báo và tạo vector
                embeddings... Quá trình này có thể mất vài chục giây.
              </span>
            </div>
          )}

          {/* Error Alert */}
          {errorMessage && (
            <div className="flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 px-3.5 py-2.5 text-xs text-rose-800">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Success Result */}
          {result && (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50/80 px-3.5 py-3 text-xs text-emerald-900">
              <div className="flex items-center gap-1.5 font-semibold text-emerald-800">
                <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                <span>Xử lý hoàn tất ({result.status})</span>
              </div>
              <div className="mt-1.5 space-y-0.5 font-mono text-[11px] text-emerald-800">
                <p>doc_id: {result.doc_id}</p>
                <p>topic_id: {result.topic_id}</p>
                <p>chunks_count: {result.chunks_count}</p>
                {result.note && <p>note: {result.note}</p>}
              </div>
            </div>
          )}

          {/* Footer Buttons */}
          <div className="flex items-center justify-end gap-2 border-t border-slate-100 pt-3">
            <button
              type="button"
              disabled={isSubmitting}
              onClick={onClose}
              className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
            >
              Đóng
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="inline-flex items-center gap-2 rounded-xl bg-blue-700 px-4 py-2 text-xs font-semibold text-white shadow-xs transition hover:bg-blue-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isSubmitting && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              <span>
                {activeTab === "arxiv" ? "Nạp từ arXiv" : "Tải lên & Lập chỉ mục"}
              </span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
