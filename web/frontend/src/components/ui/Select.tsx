import { forwardRef } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "../../lib/cn";
import "./controls.css";

export type SelectProps = React.SelectHTMLAttributes<HTMLSelectElement>;

export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { className, children, ...rest },
  ref,
) {
  return (
    <div className="ui-select-wrap">
      <select ref={ref} className={cn("ui-control", className)} {...rest}>
        {children}
      </select>
      <ChevronDown size={16} aria-hidden />
    </div>
  );
});
