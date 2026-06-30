const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

export interface ChatEvent {
  type: "tool_use" | "citation" | "sql" | "token" | "done" | "error"
  tool?: string
  input?: string
  source?: string
  page?: number
  query?: string
  content?: string
}

export async function* streamChat(
  message: string,
  history: ChatMessage[]
): AsyncGenerator<ChatEvent> {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  })

  if (!response.ok || !response.body) {
    yield { type: "error", content: "Failed to connect to backend" }
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split("\n")
    buffer = lines.pop() ?? ""
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6)) as ChatEvent
        } catch {}
      }
    }
  }
}
