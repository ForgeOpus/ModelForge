import { cn } from "../../lib/cn";
import "./Table.css";

export function Table({ className, children, ...rest }: React.TableHTMLAttributes<HTMLTableElement>) {
  return (
    <div className="ui-table-wrap">
      <table className={cn("ui-table", className)} {...rest}>
        {children}
      </table>
    </div>
  );
}

export function THead(props: React.HTMLAttributes<HTMLTableSectionElement>) {
  return <thead {...props} />;
}

export function TBody(props: React.HTMLAttributes<HTMLTableSectionElement>) {
  return <tbody {...props} />;
}

export function TR({ onClick, ...rest }: React.HTMLAttributes<HTMLTableRowElement>) {
  return <tr data-clickable={onClick ? "" : undefined} onClick={onClick} {...rest} />;
}

export function TH(props: React.ThHTMLAttributes<HTMLTableCellElement>) {
  return <th {...props} />;
}

export function TD(props: React.TdHTMLAttributes<HTMLTableCellElement>) {
  return <td {...props} />;
}
