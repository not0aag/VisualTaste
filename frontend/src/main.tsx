import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

// Bundled with the app, so the no-external-requests rule still holds.
import '@fontsource-variable/space-grotesk'
import '@fontsource-variable/jetbrains-mono'

import { App } from './App'
import './styles/index.css'

const root = document.getElementById('root')
if (!root) throw new Error('no #root element in index.html')

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
