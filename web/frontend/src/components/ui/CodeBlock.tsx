import { useState } from "react";
import { Copy, Check } from "lucide-react";
import "./CodeBlock.css";

export interface CodeBlockProps {
  code: string;
  language?: string;
}

export function CodeBlock({ code, language }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  }

  return (
    <div className="ui-codeblock">
      <button type="button" className="ui-codeblock-copy" onClick={copy} aria-label="Copy code">
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
      <pre>
        <code data-language={language}>{code}</code>
      </pre>
    </div>
  );
}
