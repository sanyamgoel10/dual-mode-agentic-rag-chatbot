"use client"

interface Citation {
  source: string
  page: number
}

interface Props {
  role: "user" | "assistant"
  content: string
  tools?: string[]
  citations?: Citation[]
  sqlQueries?: string[]
  isStreaming?: boolean
}

function ToolBadge({ tools }: { tools: string[] }) {
  if (tools.length === 0) return null
  const hasRag = tools.includes("search_docs")
  const hasSql = tools.includes("query_orders")
  const label = hasRag && hasSql ? "RAG + SQL" : hasRag ? "RAG" : "SQL"
  const colour =
    hasRag && hasSql
      ? "bg-purple-100 text-purple-700"
      : hasRag
      ? "bg-blue-100 text-blue-700"
      : "bg-green-100 text-green-700"
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${colour}`}>
      {label}
    </span>
  )
}

// Renders **bold** and *italic* markdown inline
function MarkdownText({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g)
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          return <strong key={i}>{part.slice(2, -2)}</strong>
        }
        if (part.startsWith("*") && part.endsWith("*")) {
          return <em key={i}>{part.slice(1, -1)}</em>
        }
        return <span key={i}>{part}</span>
      })}
    </>
  )
}

function CitationPill({ source, page }: Citation) {
  const name = source.replace(/\.pdf$/i, "").replace(/[_-]/g, " ")
  return (
    <div className="inline-flex items-center gap-1.5 bg-gray-50 border border-gray-200 rounded-full px-3 py-1 text-xs text-gray-600">
      <svg className="w-3 h-3 text-gray-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <span className="font-medium capitalize">{name}</span>
      <span className="text-gray-400">·</span>
      <span className="text-gray-400">p.{page}</span>
    </div>
  )
}

export default function MessageBubble({
  role,
  content,
  tools = [],
  citations = [],
  sqlQueries = [],
  isStreaming = false,
}: Props) {
  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] bg-blue-600 text-white px-4 py-2 rounded-2xl rounded-tr-sm text-sm">
          {content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] space-y-2">
        <div className="bg-white border border-gray-200 px-4 py-3 rounded-2xl rounded-tl-sm shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium text-gray-500">Assistant</span>
            <ToolBadge tools={tools} />
          </div>
          <p className="text-sm text-gray-800 whitespace-pre-wrap leading-relaxed">
            {content.split("\n").map((line, i) => (
              <span key={i}>
                {i > 0 && <br />}
                <MarkdownText text={line} />
              </span>
            ))}
            {isStreaming && (
              <span className="inline-flex items-center gap-0.5 ml-1">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="w-1 h-1 rounded-full bg-gray-400 animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </span>
            )}
          </p>
        </div>

        {citations.length > 0 && (
          <div className="flex flex-wrap gap-2 pl-1">
            {citations.map((c, i) => (
              <CitationPill key={i} source={c.source} page={c.page} />
            ))}
          </div>
        )}

        {sqlQueries.length > 0 && (
          <div className="pl-2 space-y-1">
            {sqlQueries.map((sql, i) => (
              <pre
                key={i}
                className="text-xs bg-gray-900 text-green-400 px-3 py-2 rounded-lg overflow-x-auto"
              >
                {sql}
              </pre>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
