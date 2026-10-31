/**
 * The shell: a nav bar and the six pages.
 *
 * Only Feed exists at this phase; the rest arrive in Phases 11 and 13 and are
 * listed here as placeholders so the navigation is honest about what the
 * finished app contains rather than growing a link at a time.
 */

import { NavLink, Navigate, Route, Routes, BrowserRouter } from 'react-router-dom'

import { Feed } from './pages/Feed'

const PAGES = [
  { path: '/feed', label: 'Feed' },
  { path: '/history', label: 'History' },
  { path: '/profile', label: 'Profile' },
  { path: '/lab', label: 'Model Lab' },
  { path: '/map', label: 'Embedding Map' },
  { path: '/review', label: 'Review' },
]

export function App() {
  return (
    <BrowserRouter>
      <div className="flex h-full flex-col">
        <nav className="flex shrink-0 items-center gap-1 overflow-x-auto border-b border-canvas-300 bg-canvas-100 px-3 py-1.5 dark:border-canvas-800 dark:bg-canvas-950">
          <span className="mr-3 shrink-0 text-sm font-semibold tracking-tight">
            VisualTaste
          </span>
          {PAGES.map((page) => (
            <NavLink
              key={page.path}
              to={page.path}
              className={({ isActive }) =>
                `shrink-0 rounded px-2 py-1 text-xs transition-colors ${
                  isActive
                    ? 'bg-canvas-300 text-canvas-900 dark:bg-canvas-800 dark:text-canvas-100'
                    : 'text-canvas-500 hover:text-canvas-800 dark:hover:text-canvas-200'
                }`
              }
            >
              {page.label}
            </NavLink>
          ))}
        </nav>

        <div className="min-h-0 flex-1">
          <Routes>
            <Route path="/" element={<Navigate to="/feed" replace />} />
            <Route path="/feed" element={<Feed />} />
            <Route path="/history" element={<ComingSoon name="History" phase={13} />} />
            <Route path="/profile" element={<ComingSoon name="Profile" phase={11} />} />
            <Route path="/lab" element={<ComingSoon name="Model Lab" phase={13} />} />
            <Route path="/map" element={<ComingSoon name="Embedding Map" phase={13} />} />
            <Route path="/review" element={<ComingSoon name="Review" phase={13} />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}

function ComingSoon({ name, phase }: { name: string; phase: number }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 bg-canvas-200 text-center dark:bg-canvas-900">
      <p className="font-medium">{name}</p>
      <p className="text-sm text-canvas-500">Built in phase {phase}.</p>
    </div>
  )
}
