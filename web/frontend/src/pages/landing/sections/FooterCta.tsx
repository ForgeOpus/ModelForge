import { Link } from "react-router-dom";

export function FooterCta() {
  return (
    <section className="lp-footer-cta">
      <img
        src="/modelforge-logo.jpg"
        alt=""
        aria-hidden="true"
        className="lp-footer-cta__logo"
      />
      <div className="container lp-footer-cta__inner">
        <h2>Point it at a dataset and a GPU.</h2>
        <p className="lp-footer-cta__sub">
          It handles the rest — model choice, quantization, adapters,
          parallelism.
        </p>
        <div className="cluster lp-footer-cta__ctas">
          <Link
            to="/app"
            className="ui-btn ui-btn--primary ui-btn--md shine-sweep"
          >
            Open Console
          </Link>
        </div>
      </div>
    </section>
  );
}
