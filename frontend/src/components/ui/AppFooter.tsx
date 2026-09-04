import React from 'react';

export const AppFooter: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <footer className={`py-8 px-4 text-center border-t border-black/[0.04] dark:border-white/[0.05] mt-12 ${className}`}>
      <div className="flex flex-col sm:flex-row items-center justify-center gap-2 sm:gap-4 text-xs font-mono text-slate-500 dark:text-slate-400">
        <div className="flex items-center gap-2">
          <img src="/milimo_logo.png" alt="Milimo" className="w-4 h-4 object-contain rounded" onError={(e) => { (e.target as HTMLElement).style.display = 'none'; }} />
          <span className="font-bold text-slate-700 dark:text-slate-200">Milimo Music</span>
        </div>
        <span className="hidden sm:inline text-slate-300 dark:text-slate-700">•</span>
        <span>
          Created by{' '}
          <a
            href="https://www.linkedin.com/in/mainza-kangombe-6214295"
            target="_blank"
            rel="noopener noreferrer"
            className="font-bold text-teal-600 dark:text-teal-400 hover:underline transition-colors"
          >
            Mainza Kangombe
          </a>
        </span>
        <span className="hidden sm:inline text-slate-300 dark:text-slate-700">•</span>
        <span className="text-[11px] text-slate-400 dark:text-slate-500">Production AI DAW & Studio</span>
      </div>
    </footer>
  );
};
