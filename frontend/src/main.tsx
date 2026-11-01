import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

// Bundled with the app, so the no-external-requests rule still holds.
import '@fontsource/chakra-petch/500.css'
import '@fontsource/chakra-petch/600.css'
import '@fontsource-variable/geist'
import '@fontsource-variable/geist-mono'

import { App } from './App'
import './styles/index.css'

const root = document.getElementById('root')
if (!root) throw new Error('no #root element in index.html')

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
