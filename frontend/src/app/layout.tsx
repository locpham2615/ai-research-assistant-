import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Research Assistant — Không gian Nghiên cứu Học thuật",
  description:
    "Trợ lý nghiên cứu học thuật Local-First với Citation-Grounded Agentic RAG, LangGraph, Qdrant và SQLite FTS5.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi" className="h-full antialiased">
      <body className="h-full overflow-hidden bg-slate-50 text-slate-900">
        {children}
      </body>
    </html>
  );
}
