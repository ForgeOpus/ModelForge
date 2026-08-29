import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { RunForm, type FormPayload } from "../../components/run";
import { RationaleReport } from "../../components/recommendation";
import { EmptyState } from "../../components/ui";
import { Sparkles } from "lucide-react";
import { api, ApiError } from "../../api/client";
import { useToast } from "../../components/ui/Toast";
import { useRunActivity } from "../../context/RunActivityContext";
import type { Recommendation } from "../../types/api";
import "./pages.css";

export function NewRunPage() {
  const navigate = useNavigate();
  const { push } = useToast();
  const { setActiveRunId, refreshActivity } = useRunActivity();

  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [selectedModelId, setSelectedModelId] = useState("");
  const selectRef = useRef<(id: string) => void>(() => {});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleFormState = useCallback(
    (s: {
      recommendation: Recommendation | null;
      selectedModelId: string;
      setSelectedModelId: (id: string) => void;
    }) => {
      selectRef.current = s.setSelectedModelId;
      setRecommendation(s.recommendation);
      setSelectedModelId(s.selectedModelId);
    },
    [],
  );

  async function handleSubmit(payload: FormPayload) {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const run = await api.createRun({
        use_case_slug: payload.use_case_slug,
        objective: payload.objective,
        model_id: payload.model_id,
        dataset_path: payload.dataset_path,
        dataset_remote: payload.dataset_remote,
        output_dir: payload.output_dir,
        gpu_indices: payload.gpu_indices,
        tensorboard: payload.tensorboard,
        recommendation: payload.recommendation,
      });
      setActiveRunId(run.id);
      void refreshActivity();
      navigate(`/app/runs/${run.id}`);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        push({
          title: "A run is already active",
          description: "ModelForge trains one run at a time. Open the live run or cancel it first.",
          tone: "danger",
          action: { label: "Open live run", href: "/app/live" },
        });
      } else if (err instanceof ApiError) {
        setSubmitError(err.message);
      } else {
        setSubmitError(String(err));
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="two-col">
      <div>
        <h1>New fine-tune</h1>
        <p className="muted" style={{ marginBottom: "var(--space-5)" }}>
          Pick a use case and objective, generate a hardware-aware recommendation, then launch.
        </p>
        <RunForm
          onSubmit={handleSubmit}
          submitting={submitting}
          submitError={submitError}
          onStateChange={handleFormState}
        />
      </div>
      <div className="two-col-aside">
        {recommendation ? (
          <RationaleReport
            recommendation={recommendation}
            selectedModelId={selectedModelId}
            onSelectModel={(id) => selectRef.current(id)}
          />
        ) : (
          <EmptyState
            icon={<Sparkles size={22} />}
            title="No recommendation yet"
            children={
              <>
                Generate one and this panel fills with the reasoning: the detected VRAM, the resolved
                task and strategy, the PEFT tier and its parameter-budget math, the parallelism plan,
                and the ranked candidate models.
              </>
            }
          />
        )}
      </div>
    </div>
  );
}
