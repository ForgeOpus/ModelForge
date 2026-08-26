import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

const button = cva(
  "inline-flex items-center justify-center gap-2 rounded-sm font-medium whitespace-nowrap transition-colors disabled:pointer-events-none disabled:opacity-45 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        // Brand orange is reserved for the action that commits work.
        primary: "bg-brand text-brand-ink hover:bg-brand-hi",
        secondary: "bg-raised text-text border border-line hover:border-line-strong",
        ghost: "text-dim hover:text-text hover:bg-raised",
        danger:
          "text-danger border border-danger/40 hover:border-danger hover:bg-danger/10",
        link: "text-dim hover:text-text underline-offset-4 hover:underline",
      },
      size: {
        sm: "h-7 px-2.5 text-[12px] [&_svg]:size-3.5",
        md: "h-9 px-3.5 text-[13px] [&_svg]:size-4",
        lg: "h-11 px-5 text-[14px] [&_svg]:size-4",
        icon: "size-8 [&_svg]:size-4",
      },
    },
    defaultVariants: { variant: "secondary", size: "md" },
  },
);

export function Button({
  className,
  variant,
  size,
  asChild,
  ...props
}: ComponentProps<"button"> & VariantProps<typeof button> & { asChild?: boolean }) {
  const Comp = asChild ? Slot : "button";
  return <Comp className={cn(button({ variant, size }), className)} {...props} />;
}
