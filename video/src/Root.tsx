import { Composition, registerRoot } from "remotion";
import { z } from "zod";
import { Explainer, totalFrames } from "./Explainer";
import { FPS, W, H } from "./style";

export const schema = z.object({
  repoUrl: z.string(),
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
      repoUrl: "github.com/Tar-ive/qwen-gdn-static-neuron",
    }}
  />
);

registerRoot(Root);
