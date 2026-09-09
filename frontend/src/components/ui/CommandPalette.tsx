import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
    Search,
    X,
    CornerDownLeft
} from 'lucide-react';

export interface CommandItem {
    id: string;
    title: string;
    subtitle?: string;
    category: 'Navigation' | 'Actions' | 'Transport' | 'Settings';
    icon: React.ComponentType<{ size?: number; className?: string }>;
    shortcut?: string;
    action: () => void;
}

interface CommandPaletteProps {
    isOpen: boolean;
    onClose: () => void;
    commands: CommandItem[];
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
    isOpen,
    onClose,
    commands,
}) => {
    const [query, setQuery] = useState('');
    const [selectedIndex, setSelectedIndex] = useState(0);
    const inputRef = useRef<HTMLInputElement>(null);
    const listRef = useRef<HTMLDivElement>(null);

    // Filter items based on query
    const filteredCommands = useMemo(() => {
        if (!query.trim()) return commands;
        const q = query.toLowerCase();
        return commands.filter(
            (cmd) =>
                cmd.title.toLowerCase().includes(q) ||
                cmd.category.toLowerCase().includes(q) ||
                (cmd.subtitle && cmd.subtitle.toLowerCase().includes(q))
        );
    }, [commands, query]);

    // Reset index when query changes
    useEffect(() => {
        setSelectedIndex(0);
    }, [query]);

    // Auto focus on open
    useEffect(() => {
        if (isOpen) {
            setQuery('');
            setSelectedIndex(0);
            setTimeout(() => {
                inputRef.current?.focus();
            }, 50);
        }
    }, [isOpen]);

    // Scroll selected item into view
    useEffect(() => {
        if (!listRef.current) return;
        const selectedEl = listRef.current.children[selectedIndex] as HTMLElement;
        if (selectedEl) {
            selectedEl.scrollIntoView({ block: 'nearest' });
        }
    }, [selectedIndex]);

    // Keyboard navigation
    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setSelectedIndex((prev) => (prev + 1) % Math.max(1, filteredCommands.length));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setSelectedIndex((prev) =>
                prev === 0 ? Math.max(0, filteredCommands.length - 1) : prev - 1
            );
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (filteredCommands[selectedIndex]) {
                filteredCommands[selectedIndex].action();
                onClose();
            }
        } else if (e.key === 'Escape') {
            e.preventDefault();
            onClose();
        }
    };

    if (!isOpen) return null;

    return (
        <div
            className="fixed inset-0 z-[100] flex items-start justify-center pt-20 px-4 bg-black/60 backdrop-blur-md animate-fade-in"
            onClick={onClose}
        >
            <div
                className="relative w-full max-w-2xl bg-white/95 dark:bg-[#12141c]/95 backdrop-blur-2xl rounded-2xl shadow-2xl border border-black/10 dark:border-white/10 overflow-hidden text-slate-900 dark:text-slate-100 animate-scale-up"
                onClick={(e) => e.stopPropagation()}
                onKeyDown={handleKeyDown}
            >
                {/* Search Bar */}
                <div className="flex items-center px-4 py-3.5 border-b border-black/5 dark:border-white/10 gap-3">
                    <Search size={18} className="text-teal-500 shrink-0" />
                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Type a command or search (e.g. Songs, Master, Dark, Solo)..."
                        className="w-full bg-transparent text-sm sm:text-base font-medium placeholder-slate-400 focus:outline-none"
                    />
                    {query && (
                        <button
                            onClick={() => setQuery('')}
                            className="p-1 rounded-md text-slate-400 hover:text-slate-200 transition-colors"
                        >
                            <X size={14} />
                        </button>
                    )}
                    <kbd className="hidden sm:inline-block px-2 py-0.5 text-[10px] font-mono font-bold text-slate-400 bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 rounded-md">
                        ESC
                    </kbd>
                </div>

                {/* Command List */}
                <div
                    ref={listRef}
                    className="max-h-[60vh] overflow-y-auto p-2 divide-y divide-transparent scrollbar-thin"
                >
                    {filteredCommands.length === 0 ? (
                        <div className="p-8 text-center text-slate-400 text-sm">
                            No commands matching &ldquo;{query}&rdquo;
                        </div>
                    ) : (
                        filteredCommands.map((cmd, idx) => {
                            const Icon = cmd.icon;
                            const isSelected = idx === selectedIndex;
                            return (
                                <div
                                    key={cmd.id}
                                    onClick={() => {
                                        cmd.action();
                                        onClose();
                                    }}
                                    onMouseEnter={() => setSelectedIndex(idx)}
                                    className={`flex items-center justify-between px-3.5 py-2.5 rounded-xl cursor-pointer transition-colors ${
                                        isSelected
                                            ? 'bg-teal-500/15 dark:bg-teal-500/20 text-teal-700 dark:text-teal-300 font-semibold'
                                            : 'hover:bg-black/5 dark:hover:bg-white/5 text-slate-700 dark:text-slate-300'
                                    }`}
                                >
                                    <div className="flex items-center gap-3 min-w-0">
                                        <div
                                            className={`p-2 rounded-lg ${
                                                isSelected
                                                    ? 'bg-teal-500 text-slate-950 shadow-sm'
                                                    : 'bg-black/5 dark:bg-white/5 text-slate-500 dark:text-slate-400'
                                            }`}
                                        >
                                            <Icon size={16} />
                                        </div>
                                        <div className="min-w-0">
                                            <div className="text-sm truncate font-medium">
                                                {cmd.title}
                                            </div>
                                            {cmd.subtitle && (
                                                <div className="text-xs text-slate-400 truncate">
                                                    {cmd.subtitle}
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-2 shrink-0">
                                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-black/5 dark:bg-white/5 text-slate-400 border border-black/5 dark:border-white/5">
                                            {cmd.category}
                                        </span>
                                        {cmd.shortcut && (
                                            <kbd className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 text-slate-400">
                                                {cmd.shortcut}
                                            </kbd>
                                        )}
                                        {isSelected && (
                                            <CornerDownLeft size={14} className="text-teal-500 ml-1" />
                                        )}
                                    </div>
                                </div>
                            );
                        })
                    )}
                </div>

                {/* Footer hints */}
                <div className="flex items-center justify-between px-4 py-2 border-t border-black/5 dark:border-white/10 bg-black/[0.02] dark:bg-white/[0.02] text-[11px] text-slate-400">
                    <div className="flex items-center gap-3">
                        <span>↑↓ to navigate</span>
                        <span>↵ to select</span>
                        <span>esc to close</span>
                    </div>
                    <span className="font-mono text-[10px]">⌘K</span>
                </div>
            </div>
        </div>
    );
};
