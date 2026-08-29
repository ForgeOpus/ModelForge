import { createElement } from "react";
import type { JSX } from "react";
import { cn } from "../../lib/cn";

export interface CardProps extends React.HTMLAttributes<HTMLElement> {
  interactive?: boolean;
  as?: keyof JSX.IntrinsicElements;
}

export function Card({ interactive, as = "div", className, children, ...rest }: CardProps) {
  return createElement(
    as,
    {
      className: cn("card", className),
      "data-interactive": interactive ? "" : undefined,
      ...rest,
    },
    children,
  );
}
