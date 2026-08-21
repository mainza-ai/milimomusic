import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.tsx';
import { ThemeProvider } from './context/ThemeContext.tsx';
import { AudioEngineProvider } from './context/AudioEngineContext.tsx';

window.addEventListener('unhandledrejection', (event) => {
  if (event.reason?.name === 'AbortError') {
    event.preventDefault();
  }
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <AudioEngineProvider>
        <App />
      </AudioEngineProvider>
    </ThemeProvider>
  </StrictMode>,
);
