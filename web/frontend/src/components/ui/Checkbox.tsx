import { forwardRef } from "react";
import type { ReactNode } from "react";
import { cn } from "../../lib/cn";
import "./controls.css";

export interface CheckboxProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: ReactNode;
}

export const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(function Checkbox(
  { label, className, ...rest },
  ref,
) {
  return (
    <label className={cn("ui-checkbox", className)}>
      <input ref={ref} type="checkbox" {...rest} />
      <span>{label}</span>
    </label>
  );
});
