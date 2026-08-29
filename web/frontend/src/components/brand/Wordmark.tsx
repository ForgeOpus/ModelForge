import { Link } from "react-router-dom";
import { cn } from "../../lib/cn";
import { BrandMark, type BrandMarkProps } from "./BrandMark";
import "./Wordmark.css";

export interface WordmarkProps {
  size?: "sm" | "md" | "lg";
  markVariant?: BrandMarkProps["variant"];
  showMark?: boolean;
  asLink?: boolean;
}

const MARK_SIZE: Record<NonNullable<WordmarkProps["size"]>, number> = {
  sm: 22,
  md: 28,
  lg: 40,
};

export function Wordmark({
  size = "md",
  markVariant = "svg",
  showMark = true,
  asLink = false,
}: WordmarkProps) {
  const content = (
    <>
      {showMark && <BrandMark size={MARK_SIZE[size]} variant={markVariant} />}
      <span>
        Model<span className="mf-wordmark__forge">Forge</span>
      </span>
    </>
  );

  const className = cn("mf-wordmark", `mf-wordmark--${size}`, asLink && "mf-wordmark--link");

  if (asLink) {
    return (
      <Link to="/" className={className}>
        {content}
      </Link>
    );
  }

  return <span className={className}>{content}</span>;
}
