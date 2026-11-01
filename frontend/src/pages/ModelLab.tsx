/**
 * Model Lab: what the models are doing, and the controls to change it (§7.9,
 * Part 2 §9).
 *
 * The centrepiece is one timeline with like rate, coverage and interests on
 * it together, with markers where the serving rung changed. Phase 8's lesson
 * (F5) was that a model can raise the like rate by narrowing what it shows;
 * this chart is where that becomes visible the moment it happens.
 *
 * Defaults to the REAL profile. Choosing a simulated one shows a banner that
 * stays for as long as it is selected, so a simulation is never read as Alen.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'

import { ApiError, api } from '../api/client'
import type {
  CalibrationStats,
  DiversityStats,
  Experiment,
  ExplorationStats,
  LearningCurve,
  ModelsResponse,
  ProfileListItem,
  ScoreDistribution,
  StatsSummary,
} from '../api/types'
import { Plot } from '../components/Plot'
import { binCentres, intervalText, overlaps, rungChanges, versionRow } from '../lib/labModel'

const TRAINABLE = ['r3', 'r4', 'r3+r5', 'r4+r5', 'r1', 'r2', 'r5']
const R6_MIN_LABELS = 1500

export function ModelLab() {
  const [profiles, setProfiles] = useState<ProfileListItem[]>([])
  const [profileId, setProfileId] = useState<number | null>(null)
  const [summary, setSummary] = useState<StatsSummary | null>(null)
  const [curve, setCurve] = useState<LearningCurve | null>(null)
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [scores, setScores] = useState<ScoreDistribution | null>(null)
  const [diversity, setDiversity] = useState<DiversityStats | null>(null)
  const [calibration, setCalibration] = useState<CalibrationStats | null>(null)
  const [exploration, setExploration] = useState<ExplorationStats | null>(null)
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [studies, setStudies] = useState<string[]>([])
  const [message, setMessage] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const [s, c, m, d, cal, e, x] = await Promise.all([
        api.summary(profileId),
        api.learningCurve(profileId),
        api.models(),
        api.diversity(profileId),
        api.calibration(profileId),
        api.exploration(profileId),
        api.experiments(),
      ])
      setSummary(s)
      setCurve(c)
      setModels(m)
      setDiversity(d)
      setCalibration(cal)
      setExploration(e)
      setExperiments(x)
      // The score histogram is always of the live serving model.
      setScores(profileId == null ? await api.scores() : null)
    } catch (cause) {
      setMessage(cause instanceof ApiError ? cause.detail : String(cause))
    }
  }, [profileId])

  useEffect(() => {
    void api.profiles(true).then(setProfiles).catch(() => setProfiles([]))
    void api.studies().then(setStudies).catch(() => setStudies([]))
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const act = async (label: string, run: () => Promise<unknown>) => {
    setMessage(`${label}…`)
    try {
      const result = await run()
      setMessage(`${label}: ${JSON.stringify(result).slice(0, 240)}`)
      await load()
    } catch (cause) {
      setMessage(`${label} failed: ${cause instanceof ApiError ? cause.detail : String(cause)}`)
    }
  }

  const selected = profiles.find((p) => p.profile_id === profileId)
  const simulated = summary?.is_simulated ?? selected?.is_simulated ?? false

  const timeline = useMemo(() => {
    if (!curve) return null
    const xs = curve.like_rate.map((_, i) => i + 1)
    const markers = rungChanges(curve.switches)
    return {
      data: [
        { x: xs, y: curve.like_rate, name: `like rate (window ${curve.window})`, type: 'scatter', mode: 'lines' },
        { x: xs, y: curve.coverage, name: 'coverage (last 500)', type: 'scatter', mode: 'lines' },
        { x: xs, y: curve.interests, name: 'interests (R5)', yaxis: 'y2', type: 'scatter', mode: 'lines', line: { dash: 'dot' } },
      ] as Plotly.Data[],
      layout: {
        title: { text: 'Like rate, coverage and interests, by swipe' },
        yaxis: { range: [0, 1], title: { text: 'rate' } },
        yaxis2: { overlaying: 'y', side: 'right', title: { text: 'interests' }, rangemode: 'tozero' },
        xaxis: { title: { text: 'swipe' } },
        shapes: markers.map((m) => ({
          type: 'line', x0: m.index + 1, x1: m.index + 1, yref: 'paper', y0: 0, y1: 1,
          line: { width: 1, dash: 'dash', color: '#888' },
        })),
        annotations: markers.map((m) => ({
          x: m.index + 1, y: 1, yref: 'paper', text: m.rung, showarrow: false, yanchor: 'bottom',
        })),
      } as Partial<Plotly.Layout>,
    }
  }, [curve])

  return (
    <main className="page" data-testid="model-lab">
      {simulated && (
        <div className="sticky top-0 z-10 border-b border-save/60 bg-[repeating-linear-gradient(135deg,rgba(251,191,36,0.9)_0_14px,rgba(251,191,36,0.75)_14px_28px)] px-4 py-1.5 text-center font-mono text-xs font-semibold uppercase tracking-wider text-black" role="status">
          SIMULATED profile {summary?.profile_id}: these are a planted-taste simulator's numbers, not Alen's.
        </div>
      )}
      <div className="page-inner">
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="eyebrow">Diagnostics</p>
            <h1 className="page-title mt-1">Model Lab</h1>
          </div>
          <label className="flex items-center gap-2 font-mono text-xs uppercase tracking-wider text-canvas-400">
            Profile{' '}
            <select
              className="max-w-[18rem] normal-case tracking-normal"
              value={profileId ?? ''}
              onChange={(e) => setProfileId(e.target.value === '' ? null : Number(e.target.value))}
            >
              <option value="">REAL (current)</option>
              {profiles.filter((p) => p.is_simulated).slice(0, 100).map((p) => (
                <option key={p.profile_id} value={p.profile_id}>
                  SIMULATED #{p.profile_id} · {p.labels} labels · {p.note ?? ''}
                </option>
              ))}
            </select>
          </label>
        </header>

        {message && (
          <p className="panel border-accent/30 px-3 py-2 font-mono text-[11px] text-accent/90" role="log">
            &gt; {message}
          </p>
        )}

        {summary && (
          <section className="grid grid-cols-2 gap-3 sm:grid-cols-5" aria-label="Counters">
            <Counter label="labels" value={summary.labels} />
            <Counter label="impressions" value={summary.impressions} />
            <Counter label="like rate" value={summary.like_rate?.toFixed(3) ?? '--'} />
            <Counter label="last 100" value={summary.like_rate_last_100?.toFixed(3) ?? '--'} />
            <Counter label="serving" value={`${summary.serving_rung} · ${summary.serving_version}`} />
          </section>
        )}

        {timeline && (
          <div className="panel p-3">
            <Plot data={timeline.data} layout={timeline.layout} className="h-80 w-full" testId="timeline" />
          </div>
        )}

        {models && (
          <section aria-label="Versions" className="panel overflow-x-auto p-4">
            <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
              <span className="font-medium text-white">Versions</span>
              <span className="chip">default <span className="text-accent">{models.default_rung}</span></span>
              {TRAINABLE.map((rung) => (
                <button key={rung} className="btn btn-sm font-mono"
                  onClick={() => act(`train ${rung}`, () => api.train(rung, false))}>
                  train {rung}
                </button>
              ))}
              <span className="font-mono text-[11px] text-canvas-500" title="R6 never trains inside the API.">
                R6: <code>python -m visualtaste.jobs.train_neural</code>
                {summary && summary.labels < R6_MIN_LABELS
                  ? ` (needs ${R6_MIN_LABELS} labels; this profile has ${summary.labels})`
                  : ''}
              </span>
            </div>
            <table className="w-full min-w-[720px] text-left font-mono text-xs">
              <thead className="text-[10px] uppercase tracking-wider text-canvas-500">
                <tr>
                  {['version', 'rung', 'labels', 'AUC', 'log loss', 'NDCG@10', 'ECE', 'score p95', 'λ', ''].map((h) => (
                    <th key={h} className="border-b border-white/10 px-2 py-2 font-normal">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.versions.map(versionRow).map((row) => (
                  <tr key={row.version} className={`border-b border-white/[0.04] transition-colors hover:bg-white/[0.03] ${row.active ? 'bg-accent/[0.08] text-white shadow-[inset_2px_0_0_#22d3ee]' : 'text-canvas-300'}`}>
                    <td className="px-2 py-1.5">{row.version}{row.active ? <span className="ml-1 text-accent">●</span> : ''}</td>
                    <td className="px-2 py-1.5">{row.rung}</td>
                    <td className="px-2 py-1.5">{row.labels}</td>
                    <td className="px-2 py-1.5">{row.auc}</td>
                    <td className="px-2 py-1.5">{row.logLoss}</td>
                    <td className="px-2 py-1.5">{row.ndcg}</td>
                    <td className="px-2 py-1.5">{row.ece}</td>
                    <td className="px-2 py-1.5">{row.p95}</td>
                    <td className="px-2 py-1.5">{row.lambda}</td>
                    <td className="space-x-2 px-2 py-1.5">
                      <button className="link" onClick={() => act(`evaluate ${row.version}`, () => api.evaluate(row.version))}>evaluate</button>
                      {!row.active && (
                        <button
                          className="link"
                          onClick={() => {
                            if (row.rung === 'r6' && summary && summary.labels < R6_MIN_LABELS) {
                              setMessage(
                                `Warning: R6 activated with ${summary.labels} labels, below its ${R6_MIN_LABELS} minimum.`,
                              )
                            }
                            void act(`activate ${row.version}`, () => api.activate(row.version))
                          }}
                        >
                          activate
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        <section className="grid gap-4 lg:grid-cols-2">
          {scores && (
            <Plot
              className="panel h-72 w-full p-2"
              testId="score-distribution"
              data={[
                { x: binCentres(scores.edges), y: scores.counts, type: 'bar', name: 'unseen images' },
                { x: scores.batch_scores, y: scores.batch_scores.map(() => 0), type: 'scatter', mode: 'markers', name: 'current batch', marker: { size: 10, symbol: 'triangle-up' } },
              ] as Plotly.Data[]}
              layout={{ title: { text: `Scores of unseen images (${scores.model_version})` }, bargap: 0 }}
            />
          )}
          {diversity && (
            <Plot
              className="panel h-72 w-full p-2"
              data={[
                { y: diversity.batch_diversity, name: 'batch diversity', type: 'scatter', mode: 'lines' },
                { y: diversity.batch_novelty, name: 'novelty', type: 'scatter', mode: 'lines' },
              ] as Plotly.Data[]}
              layout={{ title: { text: `Diversity and novelty per batch · coverage ${(100 * diversity.coverage_last_500).toFixed(0)}%` }, yaxis: { range: [0, 1] } }}
            />
          )}
          {calibration && calibration.samples > 0 && (
            <Plot
              className="panel h-72 w-full p-2"
              data={[
                { x: [0, 1], y: [0, 1], name: 'perfect', type: 'scatter', mode: 'lines', line: { dash: 'dot' } },
                { x: calibration.bins.map((b) => b.mean_predicted), y: calibration.bins.map((b) => b.observed_rate), text: calibration.bins.map((b) => `n = ${b.count}`), name: 'observed', type: 'scatter', mode: 'lines+markers' },
              ] as Plotly.Data[]}
              layout={{ title: { text: `Calibration · ECE ${calibration.ece?.toFixed(3) ?? '--'} · ${calibration.samples} cards` }, xaxis: { range: [0, 1], title: { text: 'predicted' } }, yaxis: { range: [0, 1], title: { text: 'observed' } } }}
            />
          )}
          {exploration && (
            <div className="panel p-4 text-sm" aria-label="Exploration yield">
              <p className="eyebrow mb-1">Bandit</p>
              <h2 className="mb-3 font-medium text-white">Exploration yield</h2>
              <p className="font-mono text-xs text-canvas-300">probe cards <span className="text-accent">{intervalText(exploration.explore)}</span></p>
              <p className="mt-1 font-mono text-xs text-canvas-300">other cards <span className="text-violet">{intervalText(exploration.exploit)}</span></p>
              <p className="mt-3 text-xs text-canvas-400">
                {overlaps(exploration.explore, exploration.exploit)
                  ? 'The intervals overlap: no detectable cost to exploring yet.'
                  : 'The intervals do not overlap.'}
              </p>
            </div>
          )}
        </section>

        {models && (
          <InterleavingPanel
            versions={models.versions.map((v) => v.version)}
            experiments={experiments}
            onAct={act}
          />
        )}

        <StudiesPanel studies={studies} onAct={act} />
      </div>
    </main>
  )
}

function Counter({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="plate p-3">
      <p className="eyebrow">{label}</p>
      <p className="mt-1.5 truncate font-display text-xl font-semibold tabular-nums text-white" title={String(value)}>{value}</p>
    </div>
  )
}

function InterleavingPanel({
  versions,
  experiments,
  onAct,
}: {
  versions: string[]
  experiments: Experiment[]
  onAct: (label: string, run: () => Promise<unknown>) => Promise<void>
}) {
  const [a, setA] = useState('')
  const [b, setB] = useState('')
  const active = experiments.find((e) => e.active)
  return (
    <section className="panel p-4 text-sm" aria-label="Interleaving">
      <p className="eyebrow mb-1">A/B</p>
      <h2 className="mb-3 font-medium text-white">Interleaving (team draft)</h2>
      {active ? (
        <div className="flex flex-wrap items-center gap-2">
          <span>Running: {active.version_a} (a) vs {active.version_b} (b)</span>
          <button className="btn btn-sm btn-danger" onClick={() => onAct('stop experiment', () => api.stopExperiment(active.experiment_id))}>stop</button>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          {[['A', a, setA], ['B', b, setB]].map(([label, value, set]) => (
            <select key={label as string} value={value as string} onChange={(e) => (set as (v: string) => void)(e.target.value)}>
              <option value="">version {label as string}…</option>
              {versions.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          ))}
          <button className="btn btn-sm" disabled={!a || !b || a === b} onClick={() => onAct('start experiment', () => api.startExperiment(a, b))}>start</button>
        </div>
      )}
      <ul className="mt-3 space-y-1 font-mono text-[11px] text-canvas-400">
        {experiments.slice(0, 8).map((e) => (
          <li key={e.experiment_id}>#{e.experiment_id}{e.active ? ' (running)' : ''}: {e.summary}</li>
        ))}
      </ul>
    </section>
  )
}

function StudiesPanel({
  studies,
  onAct,
}: {
  studies: string[]
  onAct: (label: string, run: () => Promise<unknown>) => Promise<void>
}) {
  const [study, setStudy] = useState('')
  const [seeds, setSeeds] = useState(20)
  return (
    <section className="panel p-4 text-sm" aria-label="Studies">
      <p className="eyebrow mb-1">Simulator</p>
      <h2 className="mb-3 font-medium text-white">Simulator studies</h2>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <select value={study} onChange={(e) => setStudy(e.target.value)}>
          <option value="">study…</option>
          {studies.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <label className="flex items-center gap-1 font-mono text-canvas-400">seeds <input type="number" min={1} max={50} value={seeds} onChange={(e) => setSeeds(Number(e.target.value))} className="w-16" /></label>
        <button className="btn btn-sm" disabled={!study} onClick={() => onAct(`run ${study}`, () => api.runStudy(study, seeds))}>run in the background</button>
        <span className="text-canvas-500">Results append to docs/EVALUATION.md; every seed is cached and resumable.</span>
      </div>
    </section>
  )
}
