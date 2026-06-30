"use client"
import { useState, useRef, useEffect } from "react"
import { streamChat, ChatMessage } from "@/lib/api"
import MessageBubble from "./MessageBubble"

interface DisplayMessage {
  role: "user" | "assistant"
  content: string
  tools: string[]
  citations: { source: string; page: number }[]
  sqlQueries: string[]
  isStreaming: boolean
}

export default function ChatWindow() {
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  async function handleSend() {
    if (!input.trim() || isLoading) return

    const userText = input.trim()
    setInput("")
    setIsLoading(true)

    const userMsg: DisplayMessage = {
      role: "user",
      content: userText,
      tools: [],
      citations: [],
      sqlQueries: [],
      isStreaming: false,
    }
    setMessages((prev) => [...prev, userMsg])

    const assistantMsg: DisplayMessage = {
      role: "assistant",
      content: "",
      tools: [],
      citations: [],
      sqlQueries: [],
      isStreaming: true,
    }
    setMessages((prev) => [...prev, assistantMsg])

    const history: ChatMessage[] = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }))

    try {
      for await (const event of streamChat(userText, history)) {
        setMessages((prev) => {
          const updated = [...prev]
          const last = { ...updated[updated.length - 1] }

          if (event.type === "token" && event.content) {
            last.content += event.content
          } else if (event.type === "tool_use" && event.tool) {
            if (!last.tools.includes(event.tool)) {
              last.tools = [...last.tools, event.tool]
            }
          } else if (event.type === "citation" && event.source) {
            last.citations = [
              ...last.citations,
              { source: event.source, page: event.page ?? 1 },
            ]
          } else if (event.type === "sql" && event.query) {
            last.sqlQueries = [...last.sqlQueries, event.query]
          } else if (event.type === "done") {
            last.isStreaming = false
          } else if (event.type === "error") {
            last.content = event.content ?? "An error occurred."
            last.isStreaming = false
          }

          updated[updated.length - 1] = last
          return updated
        })
      }
    } finally {
      setIsLoading(false)
      setMessages((prev) => {
        const updated = [...prev]
        const last = { ...updated[updated.length - 1] }
        last.isStreaming = false
        updated[updated.length - 1] = last
        return updated
      })
    }
  }

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto">
      <header className="px-6 py-4 border-b bg-white shadow-sm">
        <h1 className="text-lg font-semibold text-gray-900">Dual Mode Agentic RAG Chatbot</h1>
        <p className="text-xs text-gray-500">
          Ask about company policies or order data
        </p>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4 bg-gray-50">
        {messages.length === 0 && (
          <div className="text-center mt-20 space-y-2">
            <p className="text-sm text-gray-400">
              Ask a question about company policies or orders...
            </p>
            <div className="flex flex-wrap justify-center gap-2 mt-4">
              {[
                "What is the return window?",
                "How many orders are pending?",
                "What is the warranty policy?",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => setInput(q)}
                  className="text-xs border border-gray-300 rounded-full px-3 py-1 text-gray-600 hover:bg-gray-100"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble
            key={i}
            role={msg.role}
            content={msg.content}
            tools={msg.tools}
            citations={msg.citations}
            sqlQueries={msg.sqlQueries}
            isStreaming={msg.isStreaming}
          />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="px-6 py-4 border-t bg-white">
        <div className="flex gap-2">
          <input
            className="flex-1 border border-gray-300 rounded-xl px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="bg-blue-600 text-white px-4 py-2 rounded-xl text-sm font-medium disabled:opacity-50 hover:bg-blue-700 transition-colors"
          >
            {isLoading ? "..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  )
}
