import { useState, useRef, useEffect } from 'react'
import Markdown from 'react-markdown'
import { Send, SlidersHorizontal, Loader2, Bot, User } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

interface Filters {
  therapeutic_area: string
  active_substance: string
  atc_code: string
  limit: number
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
}

const DEFAULT_FILTERS: Filters = {
  therapeutic_area: '',
  active_substance: '',
  atc_code: '',
  limit: 5,
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS)
  const [showFilters, setShowFilters] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const activeFilterCount = [
    filters.therapeutic_area,
    filters.active_substance,
    filters.atc_code,
  ].filter(Boolean).length

  const sendMessage = async () => {
    const trimmed = query.trim()
    if (!trimmed || loading) return

    setQuery('')
    setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'user', content: trimmed }])
    setLoading(true)

    try {
      const res = await fetch(`${API_URL}/chats/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: trimmed,
          limit: filters.limit,
          therapeutic_area: filters.therapeutic_area || null,
          active_substance: filters.active_substance || null,
          atc_code: filters.atc_code || null,
        }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const data: string = await res.json()
      setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'assistant', content: data }])
    } catch {
      setMessages(prev => [
        ...prev,
        { id: crypto.randomUUID(), role: 'assistant', content: 'Failed to reach the server. Check your API URL and try again.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Header */}
      <header className="border-b px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
            <Bot className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-lg font-semibold leading-none">Dawa AI</h1>
            <p className="text-xs text-muted-foreground mt-0.5">EMA Pharmaceutical Reference</p>
          </div>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowFilters(v => !v)}
          className="gap-2"
        >
          <SlidersHorizontal className="h-4 w-4" />
          Filters
          {activeFilterCount > 0 && (
            <Badge className="h-5 w-5 justify-center p-0 text-[10px]">{activeFilterCount}</Badge>
          )}
        </Button>
      </header>

      {/* Filters panel */}
      {showFilters && (
        <div className="border-b bg-muted/30 px-6 py-4">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Therapeutic Area</label>
              <Input
                placeholder="e.g. oncology"
                value={filters.therapeutic_area}
                onChange={e => setFilters(f => ({ ...f, therapeutic_area: e.target.value }))}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Active Substance</label>
              <Input
                placeholder="e.g. ibuprofen"
                value={filters.active_substance}
                onChange={e => setFilters(f => ({ ...f, active_substance: e.target.value }))}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">ATC Code</label>
              <Input
                placeholder="e.g. A02BC01"
                value={filters.atc_code}
                onChange={e => setFilters(f => ({ ...f, atc_code: e.target.value }))}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Results limit</label>
              <Input
                type="number"
                min={1}
                max={20}
                value={filters.limit}
                onChange={e => setFilters(f => ({ ...f, limit: Number(e.target.value) }))}
              />
            </div>
          </div>
        </div>
      )}

      {/* Messages */}
      <ScrollArea className="flex-1">
        <div className="mx-auto max-w-3xl px-4 py-6 space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-24 text-center text-muted-foreground gap-3">
              <Bot className="h-10 w-10 opacity-30" />
              <p className="text-sm">Ask a question about an EMA medicine.</p>
            </div>
          )}

          {messages.map(msg => (
            <div
              key={msg.id}
              className={cn('flex gap-3', msg.role === 'user' && 'flex-row-reverse')}
            >
              {/* Avatar */}
              <div
                className={cn(
                  'flex h-8 w-8 shrink-0 items-center justify-center rounded-full',
                  msg.role === 'assistant' ? 'bg-primary text-primary-foreground' : 'bg-secondary',
                )}
              >
                {msg.role === 'assistant' ? (
                  <Bot className="h-4 w-4" />
                ) : (
                  <User className="h-4 w-4" />
                )}
              </div>

              {/* Bubble */}
              <div
                className={cn(
                  'max-w-[80%] rounded-2xl px-4 py-3 text-sm',
                  msg.role === 'user'
                    ? 'bg-primary text-primary-foreground rounded-tr-sm'
                    : 'bg-muted rounded-tl-sm',
                )}
              >
                {msg.role === 'assistant' ? (
                  <div className="prose-response">
                    <Markdown>{msg.content}</Markdown>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                )}
              </div>
            </div>
          ))}

          {/* Loading indicator */}
          {loading && (
            <div className="flex gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                <Bot className="h-4 w-4" />
              </div>
              <div className="flex items-center gap-2 rounded-2xl rounded-tl-sm bg-muted px-4 py-3 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Searching documents…
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="border-t bg-background px-4 py-4">
        <div className="mx-auto flex max-w-3xl gap-3">
          <Textarea
            placeholder="Ask about a medicine, dosage, contraindication…  (Enter to send, Shift+Enter for newline)"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            className="min-h-[52px] max-h-36 resize-none"
            disabled={loading}
          />
          <Button
            onClick={sendMessage}
            disabled={loading || !query.trim()}
            size="icon"
            className="h-[52px] w-[52px] shrink-0"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    </div>
  )
}
