/**
 * The shell: a glass top bar and the six pages.
 */

import { NavLink, Navigate, Route, Routes, BrowserRouter } from 'react-router-dom'

import { EmbeddingMap } from './pages/EmbeddingMap'
import { Feed } from './pages/Feed'
import { History } from './pages/History'
import { ModelLab } from './pages/ModelLab'
import { Profile } from './pages/Profile'
import { Review } from './pages/Review'

const PAGES = [
  { path: '/feed', label: 'Feed', code: '01' },
  { path: '/history', label: 'History', code: '02' },
  { path: '/profile', label: 'Profile', code: '03' },
  { path: '/lab', label: 'Model Lab', code: '04' },
  { path: '/map', label: 'Embedding Map', code: '05' },
  { path: '/review', label: 'Review', code: '06' },
]

export function App() {
  return (
    <BrowserRouter>
      <div className="flex h-full flex-col">
        <nav className="relative z-20 flex shrink-0 items-center gap-1 overflow-x-auto border-b border-white/[0.06] bg-canvas-950/70 px-3 py-2 backdrop-blur-xl sm:px-5">
          <span className="mr-4 flex shrink-0 items-center gap-2.5">
            <LogoMark />
            <span className="font-mono text-[13px] font-semibold uppercase tracking-[0.28em] text-white">
              Visual<span className="text-accent">Taste</span>
            </span>
          </span>

          {PAGES.map((page) => (
            <NavLink
              key={page.path}
              to={page.path}
              className={({ isActive }) =>
                `group relative shrink-0 rounded-md px-3 py-1.5 text-xs transition-colors ${
                  isActive
                    ? 'bg-accent/10 text-white shadow-[inset_0_0_0_1px_rgba(34,211,238,0.35)]'
                    : 'text-canvas-400 hover:bg-white/[0.04] hover:text-canvas-100'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <span
                    className={`mr-1.5 font-mono text-[9px] ${isActive ? 'text-accent' : 'text-canvas-500 group-hover:text-canvas-300'}`}
                  >
                    {page.code}
                  </span>
                  {page.label}
                  {isActive && (
                    <span className="absolute inset-x-3 -bottom-[9px] h-px bg-gradient-to-r from-transparent via-accent to-transparent" />
                  )}
                </>
              )}
            </NavLink>
          ))}

          <span
            className="ml-auto hidden shrink-0 items-center gap-2 pl-4 font-mono text-[10px] uppercase tracking-[0.2em] text-canvas-400 md:flex"
            title="Everything runs on this machine; nothing leaves it."
          >
            <span className="status-dot" aria-hidden />
            local
          </span>
        </nav>

        <div className="min-h-0 flex-1">
          <Routes>
            <Route path="/" element={<Navigate to="/feed" replace />} />
            <Route path="/feed" element={<Feed />} />
            <Route path="/history" element={<History />} />
            <Route path="/profile" element={<Profile />} />
            <Route path="/lab" element={<ModelLab />} />
            <Route path="/map" element={<EmbeddingMap />} />
            <Route path="/review" element={<Review />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}

/** An eye inside a hexagon: taste, seen by a machine. */
function LogoMark() {
  return (
    <svg viewBox="0 0 32 32" className="h-6 w-6 drop-shadow-[0_0_6px_rgba(34,211,238,0.6)]" aria-hidden>
      <defs>
        <linearGradient id="vt-logo" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#22d3ee" />
          <stop offset="1" stopColor="#a78bfa" />
        </linearGradient>
      </defs>
      <path d="M16 2 28 9v14l-12 7L4 23V9z" fill="none" stroke="url(#vt-logo)" strokeWidth="1.8" />
      <path d="M8 16s3-5 8-5 8 5 8 5-3 5-8 5-8-5-8-5z" fill="none" stroke="url(#vt-logo)" strokeWidth="1.6" />
      <circle cx="16" cy="16" r="2.4" fill="#22d3ee" />
    </svg>
  )
}
