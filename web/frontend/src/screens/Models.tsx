import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Boxes, Check, Copy, Loader2, MessagesSquare, Trash2, UploadCloud } from "lucide-react";

import { api, ApiError } from "@/lib/api";
import { formatBytes, formatRelative } from "@/lib/utils";
import type { SavedModel } from "@/types";

import { Button } from "@/components/ui/button";
import { Input, Label, Toggle } from "@/components/ui/field";
import {
  Alert, Dialog, DialogClose, DialogContent, EmptyState, Panel, Skeleton,
} from "@/components/ui/misc";
import { TierChip } from "@/components/TierChip";

export function Models() {
  const [models, setModels] = useState<SavedModel[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const load = useCallback(async () => {
    try {
      setModels(await api.listModels());
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.friendly : String(err));
      setModels([]);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (models === null) {
    return (
      <div className="flex flex-col gap-2 p-6">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-16 w-full" />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <Alert tone="danger" title="Could not load your models">{error}</Alert>
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <EmptyState
        icon={<Boxes />}
        title="No trained models yet"
        action={
          <Button variant="primary" asChild>
            <Link to="/app">Configure a fine-tune</Link>
          </Button>
        }
      >
        Every completed run is registered here, ready to chat with in the playground or push to the
        Hugging Face Hub.
      </EmptyState>
    );
  }

  return (
    <div className="flex flex-col gap-3 p-6">
      {models.map((model) => (
        <ModelCard key={model.id} model={model} onChanged={load} onChat={() => navigate("/app/playground")} />
      ))}
    </div>
  );
}

function ModelCard({
  model,
  onChanged,
  onChat,
}: {
  model: SavedModel;
  onChanged: () => void;
  onChat: () => void;
}) {
  const [copied, setCopied] = useState(false);

  async function copyPath() {
    try {
      await navigator.clipboard.writeText(model.model_path);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      toast.error("Could not copy the path.");
    }
  }

  return (
    <Panel className="p-4">
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2.5">
            <h3 className="font-display text-[15px] font-semibold">{model.name}</h3>
            <TierChip peft={model.peft_approach} />
            {Boolean(model.pushed_to_hub) && (
              <span className="font-mono text-[11px] text-ok">on the Hub</span>
            )}
          </div>
          <p className="mt-1 font-mono text-[11.5px] text-dim">
            {model.base_model ?? "—"}
            {model.task_slug && <span className="text-faint"> · {model.task_slug}</span>}
          </p>
          <p className="mt-1.5 truncate font-mono text-[11px] text-faint">{model.model_path}</p>
        </div>

        <div className="flex flex-col items-end gap-1 font-mono text-[11px] text-faint">
          <span>{formatBytes(model.size_bytes)}</span>
          <span>{formatRelative(model.created_at)}</span>
          {model.run_id != null && (
            <Link to={`/app/run/${model.run_id}`} className="hover:text-text hover:underline">
              run #{model.run_id}
            </Link>
          )}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <Button size="sm" onClick={onChat}>
          <MessagesSquare /> Open in playground
        </Button>
        <PushDialog model={model} onPushed={onChanged} />
        <Button size="sm" variant="ghost" onClick={copyPath}>
          {copied ? <Check /> : <Copy />} {copied ? "Copied" : "Copy path"}
        </Button>
        <div className="flex-1" />
        <DeleteDialog model={model} onDeleted={onChanged} />
      </div>
    </Panel>
  );
}

function PushDialog({ model, onPushed }: { model: SavedModel; onPushed: () => void }) {
  const [repo, setRepo] = useState(model.name);
  const [priv, setPriv] = useState(true);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);

  async function push() {
    setBusy(true);
    try {
      const result = await api.pushModel(model.id, repo.trim(), priv);
      toast.success(`Pushed to ${result.repo_id}`);
      setOpen(false);
      onPushed();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.friendly : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Button size="sm" variant="ghost" onClick={() => setOpen(true)}>
        <UploadCloud /> Push to Hub
      </Button>
      <DialogContent
        title="Push to the Hugging Face Hub"
        description="Uploads the whole output directory. You need to be logged in with huggingface-cli on this machine."
      >
        <Label htmlFor="repo">Repository</Label>
        <Input
          id="repo"
          value={repo}
          onChange={(e) => setRepo(e.target.value)}
          placeholder="your-username/model-name"
        />
        <label className="mt-3 flex items-center gap-2 text-[12.5px] text-dim">
          <Toggle checked={priv} onChange={setPriv} label="Private repository" />
          Keep the repository private
        </label>
        <div className="mt-5 flex justify-end gap-2">
          <DialogClose asChild>
            <Button size="sm" variant="ghost">Cancel</Button>
          </DialogClose>
          <Button size="sm" variant="primary" onClick={push} disabled={busy || !repo.trim()}>
            {busy && <Loader2 className="animate-spin" />}
            Push
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function DeleteDialog({ model, onDeleted }: { model: SavedModel; onDeleted: () => void }) {
  const [open, setOpen] = useState(false);
  const [alsoFiles, setAlsoFiles] = useState(false);
  const [busy, setBusy] = useState(false);

  async function remove() {
    setBusy(true);
    try {
      await api.deleteModel(model.id, alsoFiles);
      toast.success(alsoFiles ? "Model and files deleted" : "Model removed from the list");
      setOpen(false);
      onDeleted();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.friendly : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Button size="sm" variant="ghost" onClick={() => setOpen(true)}>
        <Trash2 /> Delete
      </Button>
      <DialogContent
        title={`Delete ${model.name}?`}
        description="This removes it from your model list. The trained files stay on disk unless you say otherwise."
      >
        <label className="flex items-start gap-2.5 rounded-sm border border-line p-3 text-[12.5px] text-dim">
          <Toggle checked={alsoFiles} onChange={setAlsoFiles} label="Also delete files from disk" />
          <span>
            Also delete <code className="font-mono text-[11.5px] text-text">{model.model_path}</code>{" "}
            from disk. This cannot be undone.
          </span>
        </label>
        <div className="mt-5 flex justify-end gap-2">
          <DialogClose asChild>
            <Button size="sm" variant="ghost">Keep it</Button>
          </DialogClose>
          <Button size="sm" variant="danger" onClick={remove} disabled={busy}>
            {busy && <Loader2 className="animate-spin" />}
            {alsoFiles ? "Delete model and files" : "Delete model"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
