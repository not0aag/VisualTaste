/**
 * Profile: what the system thinks Alen's taste is (Part 2 §7.2).
 *
 * Interest cards (six central liked images, three concept words, like count,
 * and the Beta interval of the like rate near that interest), taste in words
 * for likes and dislikes, the history of interest ids, and a reset that
 * archives the profile rather than deleting it.
 */

import { useCallback, useEffect, useState } from 'react'

import { ApiError, api } from '../api/client'
import type { ProfileResponse, SearchResponse, TasteResponse } from '../api/types'
import {
  describeHistory,
  describeInterval,
  intervalGeometry,
  sortInterests,
} from '../lib/profileModel'

export function Profile() {
  const [profile, setProfile] = useState<ProfileResponse | null>(null)
  const [taste, setTaste] = useState<TasteResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [confirming, setConfirming] = useState(false)

  const load = useCallback(async () => {
    try {
      const [p, t] = await Promise.all([api.profile(), api.taste()])
      setProfile(p)
      setTaste(t)
      setError(null)
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.detail : String(cause))
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const reset = async () => {
    setConfirming(false)
    try {
      await api.resetProfile()
      await load()
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.detail : String(cause))
    }
  }

  if (error) {
    return <p className="p-6 font-mono text-sm text-dislike">{error}</p>
  }
  if (!profile || !taste) {
    return <p className="eyebrow p-6">Loading…</p>
  }

  return (
    <main className="page">
      <div className="page-inner">
        <header className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow">Profile {profile.is_simulated ? '· simulated' : '· live'}</p>
            <h1 className="page-title mt-1">Your taste</h1>
            <div className="mt-3 flex flex-wrap gap-2">
              <Stat label="likes" value={taste.likes} tone="text-like" />
              <Stat label="dislikes" value={taste.dislikes} tone="text-dislike" />
              <Stat label="labels" value={profile.labels} tone="text-white" />
              <Stat label="serving" value={(profile.active_rung ?? 'r0').toUpperCase()} tone="text-accent" />
              {profile.is_simulated && <span className="chip border-save/50 text-save">SIMULATED profile</span>}
            </div>
          </div>
          {confirming ? (
            <div className="panel flex flex-wrap items-center gap-2 border-dislike/40 px-3 py-2 text-sm" role="alertdialog">
              <span className="text-canvas-200">Start a fresh profile? The current one is archived, not deleted.</span>
              <button className="btn btn-sm btn-danger" onClick={reset}>
                Reset
              </button>
              <button className="btn btn-sm" onClick={() => setConfirming(false)}>
                Cancel
              </button>
            </div>
          ) : (
            <button className="btn" onClick={() => setConfirming(true)}>
              Reset profile…
            </button>
          )}
        </header>

        {taste.note && <p className="panel px-4 py-3 text-sm text-canvas-300">{taste.note}</p>}

        <section aria-label="Interests" className="grid gap-4 sm:grid-cols-2">
          {sortInterests(taste.interests).map((card) => {
            const bar = intervalGeometry(card.like_rate)
            return (
              <article key={card.interest_id} className="plate p-4 transition-all hover:border-accent/30 hover:shadow-glow">
                <div className="flex items-baseline justify-between gap-2">
                  <h2 className="truncate text-base font-medium text-white">
                    <span className="mr-2 font-mono text-xs text-accent">#{card.interest_id}</span>
                    {card.label || 'unnamed interest'}
                  </h2>
                  <span className="chip shrink-0">{card.n_likes} likes</span>
                </div>
                <div className="mt-3 grid grid-cols-6 gap-1.5">
                  {card.central_image_ids.map((id) => (
                    <img key={id} src={api.thumbUrl(id)} alt="" loading="lazy" className="thumb hover:scale-105" />
                  ))}
                </div>
                <div className="mt-4" title={describeInterval(card.like_rate)}>
                  <div className="relative h-1.5 rounded-full bg-canvas-700">
                    <div
                      className="absolute h-1.5 rounded-full bg-gradient-to-r from-accent to-violet shadow-[0_0_10px_rgba(34,211,238,0.5)]"
                      style={{ left: `${bar.left}%`, width: `${bar.width}%` }}
                    />
                    <div
                      className="absolute -top-1 h-3.5 w-0.5 rounded bg-white shadow-[0_0_6px_white]"
                      style={{ left: `${bar.mean}%` }}
                    />
                  </div>
                  <p className="mt-2 font-mono text-[11px] text-canvas-400">{describeInterval(card.like_rate)}</p>
                </div>
                <MoreOfInterest imageIds={card.central_image_ids} />
              </article>
            )
          })}
        </section>

        <section className="grid gap-4 sm:grid-cols-2">
          <WordList title="What you like, in words" words={taste.likes_in_words} tone="like" />
          <WordList title="What you dislike, in words" words={taste.dislikes_in_words} tone="dislike" />
        </section>

        {taste.history.length > 0 && (
          <section className="panel p-4">
            <p className="eyebrow mb-2">Timeline</p>
            <h2 className="mb-3 text-sm font-medium text-white">How the interests have changed</h2>
            <ol className="space-y-1.5 border-l border-accent/30 pl-4 text-sm text-canvas-300">
              {describeHistory(taste.history).map((line) => (
                <li key={line} className="relative">
                  <span className="absolute -left-[21px] top-1.5 h-2 w-2 rounded-full bg-accent shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
                  {line}
                </li>
              ))}
            </ol>
            <p className="mt-3 font-mono text-[11px] text-canvas-500">
              Ids stay stable across reclusters (Hungarian matching); source: {taste.source}.
            </p>
          </section>
        )}
      </div>
    </main>
  )
}

/** Unrated images near this interest's centre, on request. */
function MoreOfInterest({ imageIds }: { imageIds: string[] }) {
  const [results, setResults] = useState<SearchResponse['results'] | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  if (imageIds.length === 0) return null

  const load = async () => {
    setBusy(true)
    try {
      setResults((await api.similarTo(imageIds, 12)).results)
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.detail : String(cause))
    } finally {
      setBusy(false)
    }
  }

  if (!results) {
    return (
      <div className="mt-3 flex items-center gap-3">
        <button className="btn btn-sm" onClick={() => void load()} disabled={busy}>
          {busy ? 'Finding…' : 'Show more like this'}
        </button>
        {error && <span className="font-mono text-[11px] text-dislike">{error}</span>}
      </div>
    )
  }

  return (
    <div className="mt-3" data-testid="more-of-interest">
      <p className="eyebrow mb-1.5 text-canvas-400">Not yet rated, closest to this interest</p>
      <div className="grid grid-cols-6 gap-1.5">
        {results.map(({ image }) => (
          <img key={image.image_id} src={image.thumb_url} alt="" loading="lazy" className="thumb" />
        ))}
      </div>
    </div>
  )
}

function Stat({ label, value, tone }: { label: string; value: string | number; tone: string }) {
  return (
    <span className="chip">
      <span className={`font-semibold ${tone}`}>{value}</span>
      <span className="text-canvas-500">{label}</span>
    </span>
  )
}

function WordList({
  title,
  words,
  tone,
}: {
  title: string
  words: TasteResponse['likes_in_words']
  tone: 'like' | 'dislike'
}) {
  const chipTone =
    tone === 'like'
      ? 'border-like/30 bg-like/10 text-like'
      : 'border-dislike/30 bg-dislike/10 text-dislike'
  return (
    <div className="panel p-4">
      <h2 className="mb-3 text-sm font-medium text-white">{title}</h2>
      {words.length === 0 ? (
        <p className="font-mono text-xs text-canvas-500">Not enough ratings yet.</p>
      ) : (
        <ul className="flex flex-wrap gap-1.5">
          {words.map((word) => (
            <li
              key={word.concept}
              className={`rounded-full border px-2.5 py-0.5 font-mono text-[11px] ${chipTone}`}
              title={`z = ${word.z_score.toFixed(2)} against the vocabulary's mean similarity`}
            >
              {word.concept}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
