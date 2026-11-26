import { Composition, registerRoot } from "remotion";
import { z } from "zod";
import { Explainer, totalFrames } from "./Explainer";
import { FPS, W, H } from "./style";

export const schema = z.object({
  resultNote: z.string(),
  repoUrl: z.string(),
  costNote: z.string(),
});

const Root = () => (
  <Composition
    id="Explainer"
    component={Explainer}
    schema={schema}
    durationInFrames={totalFrames()}
    fps={FPS}
    width={W}
    height={H}
    defaultProps={{
      resultNote: "Measured: Qwen3.5-0.8B rehearsal on 2 NeuronCores (same code). Qwen3.8-27B run pending.",
      repoUrl: "github.com/Tar-ive/qwen-gdn-static-neuron",
      costNote: "trn1.32xlarge Spot, us-east-2c, Nov 2026 · OpenRouter cheapest Qwen3.8-27B output price",
    }}
  />
);

registerRoot(Root);
