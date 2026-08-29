import {
  RationaleReport,
  sampleRecommendation,
} from "../../../components/recommendation";

export function RationaleShowcase() {
  return (
    <section id="rationale" className="lp-section">
      <div className="container">
        <div className="lp-head">
          <h2>The rationale is the product.</h2>
          <p>
            Other tools give you a config. ModelForge gives you the config and
            the argument for it.
          </p>
        </div>
        <div className="lp-rationale__frame card">
          <RationaleReport recommendation={sampleRecommendation} compact />
        </div>
      </div>
    </section>
  );
}
