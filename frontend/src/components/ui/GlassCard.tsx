import React, { type ReactNode } from 'react';
import { motion } from 'framer-motion';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface GlassCardProps {
    children: ReactNode;
    className?: string;
    delay?: number;
    onClick?: () => void;
}

export const GlassCard: React.FC<GlassCardProps> = ({ children, className, delay = 0, onClick }) => {
    return (
        <motion.div
            onClick={onClick}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay, ease: "easeOut" }}
            className={cn(
                "rounded-2xl p-6 border",
                "bg-white/80 dark:bg-[#141620]/85 backdrop-blur-2xl",
                "border-black/[0.06] dark:border-white/[0.08]",
                "shadow-apple-sm dark:shadow-2xl text-slate-900 dark:text-slate-100",
                "hover:shadow-apple-md transition-all duration-300",
                className
            )}
        >
            {children}
        </motion.div>
    );
};
