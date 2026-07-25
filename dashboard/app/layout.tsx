import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AgentOS-Lite — Kernel Dashboard",
  description: "Live view of the LLM-agent operating system kernel",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-mono antialiased">{children}</body>
    </html>
  );
}
