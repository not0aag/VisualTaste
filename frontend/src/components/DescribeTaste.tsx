/**
 * "Describe what you like": shown on the Feed during cold start (ROADMAP D4).
 *
 * A few words aim three of every five cold-start cards at the description,
 * until enough verdicts exist for a learned model to take over. Optional, and
 * one click to forget.
 */

import { useEffect, useState } from 'react'

import { ApiError, api } from '../api/client'

export function DescribeTaste({ onChange }: { onChange?: () => void }) {
  const [saved, setSaved] = useState<string | null | undefined>(undefined)
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .seedText()
      .then((response) => setSaved(response.text))
      .catch(() => setSaved(null)) // unavailable (e.g. the online demo): hide quietly
  }, [])

  const save = async () => {
    if (!draft.trim()) return
    setBusy(true)
    setError(null)
    try {
      setSaved((await api.setSeedText(draft.trim())).text)
      setDraft('')
      onChange?.()
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.detail : String(cause))
    } finally {
      setBusy(false)
    }
  }

  const forget = async () => {
    await api.clearSeedText().catch(() => undefined)
    setSaved(null)
    onChange?.()
  }

  if (saved === undefined) return null

  if (saved) {
    return (
      <p className="chip mx-auto" data-testid="described-taste">
        Starting from <span className="text-accent">“{saved}”</span>
        <button className="ml-1 text-canvas-400 hover:text-white" onClick={() => void forget()} aria-label="Forget the description">
          ✕
        </button>
      </p>
    )
  }

  return (
    <form
      className="mx-auto flex w-full max-w-md items-center gap-2"
      data-testid="describe-taste"
      onSubmit={(event) => {
        event.preventDefault()
        void save()
      }}
    >
      <label htmlFor="describe-taste" className="sr-only">
        Describe what you like, to start from there
      </label>
      <input
        id="describe-taste"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        placeholder="Optional: describe what you like, e.g. “foggy forests, old maps”"
        maxLength={200}
        className="min-w-0 flex-1 rounded-plate border border-white/10 bg-canvas-900/70 px-3 py-1.5 text-xs text-white outline-none placeholder:text-canvas-500 focus:border-accent/60"
      />
      <button type="submit" className="btn btn-sm" disabled={busy || !draft.trim()}>
        {busy ? 'Loading…' : 'Start there'}
      </button>
      {error && <span className="font-mono text-[11px] text-dislike">{error}</span>}
    </form>
  )
}
