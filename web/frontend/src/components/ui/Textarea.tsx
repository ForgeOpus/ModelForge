import { forwardRef } from "react";
import { cn } from "../../lib/cn";
import "./controls.css";

export type TextareaProps = React.TextareaHTMLAttributes<HTMLTextAreaElement>;

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(function Textarea(
  { className, ...rest },
  ref,
) {
  return <textarea ref={ref} className={cn("ui-control", "ui-textarea", className)} {...rest} />;
});
