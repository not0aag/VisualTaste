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
import type { ProfileResponse, TasteResponse } from '../api/types'
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
    return <p className="p-6 text-sm text-red-600 dark:text-red-400">{error}</p>
  }
  if (!profile || !taste) {
    return <p className="p-6 text-sm text-canvas-500">Loading…</p>
  }

  return (
    <main className="h-full overflow-y-auto bg-canvas-200 dark:bg-canvas-900">
      <div className="mx-auto flex max-w-5xl flex-col gap-6 p-4 sm:p-6">
        <header className="flex flex-wrap items-baseline justify-between gap-3">
          <div>
            <h1 className="text-lg font-semibold">Your taste</h1>
            <p className="text-sm text-canvas-500">
              {taste.likes} likes, {taste.dislikes} dislikes, {profile.labels} labels in all ·
              serving {profile.active_rung ?? 'r0'}
              {profile.is_simulated ? ' · SIMULATED profile' : ''}
            </p>
          </div>
          {confirming ? (
            <div className="flex items-center gap-2 text-sm" role="alertdialog">
              <span>Start a fresh profile? The current one is archived, not deleted.</span>
              <button className="rounded bg-red-600 px-3 py-1 text-white" onClick={reset}>
                Reset
              </button>
              <button className="rounded px-3 py-1" onClick={() => setConfirming(false)}>
                Cancel
              </button>
            </div>
          ) : (
            <button
              className="rounded border border-canvas-400 px-3 py-1 text-sm"
              onClick={() => setConfirming(true)}
            >
              Reset profile…
            </button>
          )}
        </header>

        {taste.note && <p className="text-sm text-canvas-500">{taste.note}</p>}

        <section aria-label="Interests" className="grid gap-4 sm:grid-cols-2">
          {sortInterests(taste.interests).map((card) => {
            const bar = intervalGeometry(card.like_rate)
            return (
              <article
                key={card.interest_id}
                className="rounded-lg bg-canvas-100 p-3 shadow-sm dark:bg-canvas-950"
              >
                <h2 className="text-sm font-medium">
                  #{card.interest_id} · {card.label || 'unnamed interest'}
                </h2>
                <p className="text-xs text-canvas-500">{card.n_likes} likes</p>
                <div className="mt-2 grid grid-cols-6 gap-1">
                  {card.central_image_ids.map((id) => (
                    <img
                      key={id}
                      src={api.thumbUrl(id)}
                      alt=""
                      loading="lazy"
                      className="aspect-square w-full rounded object-cover"
                    />
                  ))}
                </div>
                <div className="mt-3" title={describeInterval(card.like_rate)}>
                  <div className="relative h-2 rounded bg-canvas-300 dark:bg-canvas-800">
                    <div
                      className="absolute h-2 rounded bg-emerald-500/60"
                      style={{ left: `${bar.left}%`, width: `${bar.width}%` }}
                    />
                    <div
                      className="absolute -top-0.5 h-3 w-0.5 bg-emerald-700"
                      style={{ left: `${bar.mean}%` }}
                    />
                  </div>
                  <p className="mt-1 text-xs text-canvas-500">{describeInterval(card.like_rate)}</p>
                </div>
              </article>
            )
          })}
        </section>

        <section className="grid gap-4 sm:grid-cols-2">
          <WordList title="What you like, in words" words={taste.likes_in_words} />
          <WordList title="What you dislike, in words" words={taste.dislikes_in_words} />
        </section>

        {taste.history.length > 0 && (
          <section>
            <h2 className="mb-1 text-sm font-medium">How the interests have changed</h2>
            <ol className="list-inside list-decimal text-sm text-canvas-600 dark:text-canvas-400">
              {describeHistory(taste.history).map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ol>
            <p className="mt-1 text-xs text-canvas-500">
              Ids stay stable across reclusters (Hungarian matching); source: {taste.source}.
            </p>
          </section>
        )}
      </div>
    </main>
  )
}

function WordList({ title, words }: { title: string; words: TasteResponse['likes_in_words'] }) {
  return (
    <div className="rounded-lg bg-canvas-100 p-3 dark:bg-canvas-950">
      <h2 className="mb-1 text-sm font-medium">{title}</h2>
      {words.length === 0 ? (
        <p className="text-xs text-canvas-500">Not enough ratings yet.</p>
      ) : (
        <ul className="flex flex-wrap gap-1">
          {words.map((word) => (
            <li
              key={word.concept}
              className="rounded bg-canvas-200 px-2 py-0.5 text-xs dark:bg-canvas-800"
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
