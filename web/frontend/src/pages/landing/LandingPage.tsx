import "./landing.css";
import { Hero } from "./sections/Hero";
import { HowItWorks } from "./sections/HowItWorks";
import { RationaleShowcase } from "./sections/RationaleShowcase";
import { FeatureGrid } from "./sections/FeatureGrid";
import { FooterCta } from "./sections/FooterCta";

export function LandingPage() {
  return (
    <>
      <Hero />
      <HowItWorks />
      <RationaleShowcase />
      <FeatureGrid />
      <FooterCta />
    </>
  );
}
