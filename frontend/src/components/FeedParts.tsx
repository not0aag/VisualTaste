/**
 * The Feed's instrument parts: ambient light, verdict flash, the learning
 * ring, a rolling counter, and the verdict buttons that echo the keyboard.
 */

import { AnimatePresence, animate, motion, useReducedMotion } from 'framer-motion'
import { useEffect, useRef } from 'react'

import type { Action } from '../api/types'

const VERDICT_GLOW: Record<Action, string> = {
  like: 'rgba(61,220,151,0.55)',
  dislike: 'rgba(255,107,129,0.55)',
  save: 'rgba(255,193,69,0.55)',
  skip: 'rgba(163,178,198,0.35)',
}

/**
 * The room takes on the photograph's light: its two most dominant colours,
 * softly, behind everything. Crossfades from card to card.
 */
export function AmbientLight({ colours, imageId }: { colours: string[]; imageId: string }) {
  const a = colours[0] ?? '#3fd8f0'
  const b = colours[1] ?? a
  return (
    <AnimatePresence initial={false}>
      <motion.div
        key={imageId}
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-10"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.9, ease: 'easeOut' }}
        style={{
          background: `radial-gradient(48rem 30rem at 50% 42%, ${a}30, transparent 70%), radial-gradient(40rem 26rem at 85% 95%, ${b}26, transparent 70%)`,
        }}
      />
    </AnimatePresence>
  )
}

/** A brief inner glow at the screen's edge, in the colour of the verdict. */
export function EdgeFlash({ flash }: { flash: { action: Action; n: number } | null }) {
  const reduce = useReducedMotion()
  if (!flash || reduce) return null
  return (
    <motion.div
      key={flash.n}
      aria-hidden
      className="pointer-events-none absolute inset-0 z-30"
      initial={{ opacity: 1 }}
      animate={{ opacity: 0 }}
      transition={{ duration: 0.55, ease: 'easeOut' }}
      style={{ boxShadow: `inset 0 0 140px -20px ${VERDICT_GLOW[flash.action]}` }}
    />
  )
}

/** Progress toward the next online retrain, as a small ring with its label. */
export function LearningRing({
  done,
  total,
  label,
  busy,
}: {
  done: number
  total: number
  label: string
  busy: boolean
}) {
  const r = 9
  const circumference = 2 * Math.PI * r
  return (
    <span className="hidden items-center gap-2 sm:flex" title={label} data-testid="learning-ring">
      <svg viewBox="0 0 24 24" className={`h-6 w-6 -rotate-90 ${busy ? 'animate-spin-slow' : ''}`} aria-hidden>
        <circle cx="12" cy="12" r={r} className="fill-none stroke-white/10" strokeWidth="2.5" />
        <motion.circle
          cx="12"
          cy="12"
          r={r}
          className="fill-none stroke-violet"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={false}
          animate={{ strokeDashoffset: busy ? circumference * 0.7 : circumference * (1 - done / total) }}
          transition={{ type: 'spring', stiffness: 160, damping: 24 }}
          style={{ filter: 'drop-shadow(0 0 3px rgba(165,148,249,0.8))' }}
        />
      </svg>
      <span className="font-mono text-[11px] text-canvas-400">{label}</span>
    </span>
  )
}

/** A count that rolls to its new value instead of jumping. */
export function AnimatedNumber({ value, className }: { value: number; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null)
  const shown = useRef(value)
  const reduce = useReducedMotion()

  useEffect(() => {
    const element = ref.current
    if (!element) return
    if (reduce) {
      element.textContent = value.toLocaleString()
      shown.current = value
      return
    }
    const controls = animate(shown.current, value, {
      duration: 0.35,
      ease: 'easeOut',
      onUpdate: (latest) => {
        shown.current = latest
        element.textContent = Math.round(latest).toLocaleString()
      },
    })
    return () => controls.stop()
  }, [value, reduce])

  return (
    <span ref={ref} className={className}>
      {value.toLocaleString()}
    </span>
  )
}

export function Centred({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full flex-col items-center justify-center p-8 text-center">
      <div className="plate flex animate-fade-up flex-col items-center px-10 py-8">{children}</div>
    </div>
  )
}

export function Spinner() {
  return (
    <span className="relative inline-flex h-10 w-10" aria-hidden>
      <span className="absolute inset-0 rounded-full border border-white/10" />
      <span className="absolute inset-0 animate-spin-slow rounded-full border-2 border-transparent border-t-accent shadow-[0_0_14px_-2px_rgba(63,216,240,0.7)]" />
    </span>
  )
}

const TONES = {
  like: {
    base: 'border-like/40 text-like hover:bg-like/15 hover:shadow-glow-like',
    on: 'bg-like/20 shadow-glow-like',
  },
  dislike: {
    base: 'border-dislike/40 text-dislike hover:bg-dislike/15 hover:shadow-glow-dislike',
    on: 'bg-dislike/20 shadow-glow-dislike',
  },
  save: {
    base: 'border-save/40 text-save hover:bg-save/15 hover:shadow-glow-save',
    on: 'bg-save/20 shadow-glow-save',
  },
  plain: {
    base: 'border-white/10 text-canvas-300 hover:border-accent/40 hover:bg-accent/10 hover:text-white',
    on: 'border-accent/50 bg-accent/15 text-white shadow-glow',
  },
}

export function VerdictButton({
  children,
  onClick,
  testId,
  tone,
  disabled,
  keyHint,
  pressed,
}: {
  children: React.ReactNode
  onClick: () => void
  testId: string
  tone?: 'like' | 'dislike' | 'save'
  disabled?: boolean
  keyHint?: string
  /** True for a moment after the matching key is pressed: the key echoes on screen. */
  pressed?: boolean
}) {
  const colours = TONES[tone ?? 'plain']
  return (
    <button
      type="button"
      data-testid={testId}
      onClick={onClick}
      disabled={disabled}
      className={`group inline-flex items-center gap-2 rounded-plate border bg-canvas-900/70 px-4 py-2 text-sm font-medium backdrop-blur transition-all duration-150 active:scale-95 disabled:cursor-not-allowed disabled:opacity-30 disabled:shadow-none ${colours.base} ${pressed ? `scale-95 ${colours.on}` : ''}`}
    >
      {keyHint && (
        <span
          className={`kbd transition-transform ${pressed ? 'translate-y-px opacity-100 shadow-none' : 'opacity-70 group-hover:opacity-100'}`}
          aria-hidden
        >
          {keyHint}
        </span>
      )}
      {children}
    </button>
  )
}
