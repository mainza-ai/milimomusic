import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Check, Search, X } from 'lucide-react';

interface ComboboxProps {
    value: string;
    onChange: (value: string) => void;
    options: string[];
    placeholder?: string;
    label?: string;
    className?: string;
    disabled?: boolean;
    onRefresh?: () => void;
    isLoading?: boolean;
}

export const Combobox: React.FC<ComboboxProps> = ({
    value,
    onChange,
    options,
    placeholder = "Select option...",
    label,
    className = "",
    disabled = false,
    onRefresh,
    isLoading = false
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const [searchTerm, setSearchTerm] = useState("");
    const wrapperRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // Filter options based on search
    // If value is not in options, we still want to show it as selected or allow custom input?
    // User requested "dropdown list... specific to each provider", but usually advanced users want to type custom
    // Let's allow custom input by treating the search term as the value if entered

    const filteredOptions = options.filter(opt =>
        opt.toLowerCase().includes(searchTerm.toLowerCase())
    );

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    // Sync internal search state with external value when closed
    useEffect(() => {
        if (!isOpen) {
            // If we closed, what does the input show? the value.
            // But if it's currently open, we want the search term.
        }
    }, [isOpen, value]);

    const handleSelect = (option: string) => {
        onChange(option);
        setIsOpen(false);
        setSearchTerm("");
    };

    const handleCustomSubmit = () => {
        if (searchTerm) {
            onChange(searchTerm);
            setIsOpen(false);
        }
    };

    return (
        <div className={`relative ${className}`} ref={wrapperRef}>
            {label && <label className="text-xs font-bold text-slate-500 uppercase mb-1 block">{label}</label>}

            <div
                className={`
                    w-full px-3 py-2 border border-slate-200 rounded-md bg-white 
                    flex items-center justify-between gap-2 cursor-pointer 
                    focus-within:ring-2 focus-within:ring-cyan-500 transition-all
                    ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-slate-300'}
                `}
                onClick={() => !disabled && setIsOpen(prev => !prev)}
            >
                {/* Main Display: either the selected value or placeholder */}
                <span className={`text-sm font-mono truncate ${value ? 'text-slate-800' : 'text-slate-400'}`}>
                    {value || placeholder}
                </span>

                <div className="flex items-center gap-1">
                    {value && !disabled && (
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                onChange("");
                            }}
                            className="p-1 hover:bg-slate-100 rounded-full text-slate-400"
                        >
                            <X className="w-3 h-3" />
                        </button>
                    )}
                    <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                </div>
            </div>

            <AnimatePresence>
                {isOpen && !disabled && (
                    <motion.div
                        initial={{ opacity: 0, y: -5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -5 }}
                        className="absolute z-50 w-full mt-1 bg-white border border-slate-200 rounded-md shadow-lg overflow-hidden max-h-60 flex flex-col"
                    >
                        {/* Search Input */}
                        <div className="p-2 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
                            <Search className="w-3.5 h-3.5 text-slate-400" />
                            <input
                                ref={inputRef}
                                autoFocus
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter') handleCustomSubmit();
                                }}
                                className="bg-transparent border-none outline-none text-xs w-full text-slate-700 font-mono placeholder:text-slate-400"
                                placeholder="Search or type custom..."
                                onClick={(e) => e.stopPropagation()}
                            />
                        </div>

                        {/* List */}
                        <div className="overflow-y-auto flex-1 p-1">
                            {filteredOptions.length > 0 ? (
                                filteredOptions.map(option => (
                                    <button
                                        key={option}
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            handleSelect(option);
                                        }}
                                        className={`
                                            w-full text-left px-3 py-2 text-xs font-mono rounded-sm flex items-center justify-between
                                            ${value === option ? 'bg-cyan-50 text-cyan-700' : 'text-slate-600 hover:bg-slate-50'}
                                        `}
                                    >
                                        <span className="truncate">{option}</span>
                                        {value === option && <Check className="w-3 h-3" />}
                                    </button>
                                ))
                            ) : (
                                <div className="p-4 text-center text-xs text-slate-400 italic">
                                    {searchTerm ? (
                                        <button
                                            onClick={handleCustomSubmit}
                                            className="text-cyan-600 hover:underline"
                                        >
                                            Use "{searchTerm}"
                                        </button>
                                    ) : (
                                        onRefresh ? "No models found." : "No options."
                                    )}
                                </div>
                            )}
                        </div>

                        {/* Footer (Refresh Action) */}
                        {onRefresh && (
                            <div className="border-t border-slate-100 p-1 bg-slate-50">
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onRefresh();
                                    }}
                                    className="w-full py-2 text-xs text-center text-cyan-600 hover:text-cyan-800 hover:bg-cyan-50/50 rounded flex items-center justify-center gap-1 font-medium transition-colors"
                                    disabled={isLoading}
                                >
                                    {isLoading ? <span className="animate-spin">⌛</span> : "↻ Refresh List"}
                                </button>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
