import { forwardRef } from "react";
import { cn } from "../../lib/cn";
import "./controls.css";

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { className, type = "text", ...rest },
  ref,
) {
  return <input ref={ref} type={type} className={cn("ui-control", className)} {...rest} />;
});
