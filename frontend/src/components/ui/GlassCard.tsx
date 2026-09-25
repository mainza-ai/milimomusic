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
    animateEntry?: boolean;
}

export const GlassCard: React.FC<GlassCardProps> = ({
    children,
    className,
    delay = 0,
    onClick,
    animateEntry = true
}) => {
    const baseClasses = cn(
        "rounded-2xl p-6 border",
        "bg-white/95 dark:bg-[#141620]/95",
        "border-black/[0.06] dark:border-white/[0.08]",
        "shadow-apple-sm dark:shadow-2xl text-slate-900 dark:text-slate-100",
        "hover:shadow-apple-md transition-shadow duration-200",
        "transform-gpu",
        className
    );

    if (!animateEntry) {
        return (
            <div onClick={onClick} className={baseClasses}>
                {children}
            </div>
        );
    }

    return (
        <motion.div
            onClick={onClick}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay, ease: "easeOut" }}
            className={baseClasses}
        >
            {children}
        </motion.div>
    );
};
