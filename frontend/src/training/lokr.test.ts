import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import {
  applyReadonlyDefaults,
  cloneFormModel,
  createDefaultModel,
  executeSchemaSources,
  normalizeModelForSchema,
  serializeModel,
  type FormModel,
} from "../schema/adapter"
import { buildTrainingConfig, hydrateImportedConfig } from "./params"

function loadSchema(name: string) {
  const read = (file: string) => readFileSync(resolve(process.cwd(), `../mikazuki/schema/${file}.ts`), "utf8")
  return executeSchemaSources(
    [
      { name: "shared", hash: "test", schema: read("shared") },
      { name, hash: "test", schema: read(name) },
    ],
    name,
  )
}

const schema = loadSchema("sd3-lora")

/** Mirrors TrainingPage: `output = buildTrainingConfig(serializeModel(schema, model))`. */
function exportedConfig(model: FormModel) {
  return buildTrainingConfig(serializeModel(schema, model), "sd3-lora")
}

/** Mirrors TrainingPage.applyImportedConfig / the autosave restore steps. */
function importIntoForm(config: FormModel) {
  const defaults = createDefaultModel(schema)
  const explicitKeys = new Set(Object.keys(config))
  const imported = hydrateImportedConfig(cloneFormModel(config))
  const model = normalizeModelForSchema(schema, { ...cloneFormModel(defaults), ...imported }, { explicitKeys })
  applyReadonlyDefaults(schema, model, defaults)
  return model
}

function lokrModel(): FormModel {
  return {
    ...createDefaultModel(schema),
    lora_type: "lokr",
    lokr_factor: 16,
    full_matrix: true,
    use_cp: true,
    conv_dim: 8,
    conv_alpha: 8,
    network_dim: 16,
    network_alpha: 16,
  }
}

describe("LoKr parameters survive a config round trip", () => {
  it("writes the lycoris network_args from the active branch", () => {
    const serialized = serializeModel(schema, lokrModel())
    const config = buildTrainingConfig(serialized, "sd3-lora")

    expect(serialized.lokr_factor).toBe(16)
    expect(config.lora_type).toBe("lokr")
    expect(config.network_module).toBe("lycoris.kohya")
    expect(config.network_args).toEqual(expect.arrayContaining(["algo=lokr", "factor=16"]))
    // lokr_factor is folded into network_args; the remaining LyCORIS switches
    // stay top-level (the backend adapter injects them after this step).
    expect(config).not.toHaveProperty("lokr_factor")
    expect(config.full_matrix).toBe(true)
    expect(config.use_cp).toBe(true)
  })

  it("rehydrates lycoris network_args into typed UI fields", () => {
    const imported = hydrateImportedConfig({
      network_module: "lycoris.kohya",
      network_args: [
        "conv_dim=4",
        "algo=lokr",
        "factor=16",
        "use_cp=True",
        "train_norm=False",
        "dora_wd=True",
        "rank_dropout=0.5",
        "unit=8",
        "module_dropout=0.1",
        "dropout=undefined",
      ],
    })

    expect(imported.lycoris_algo).toBe("lokr")
    expect(imported.lokr_factor).toBe(16)
    expect(imported.conv_dim).toBe(4)
    expect(imported.use_cp).toBe(true)
    expect(imported.train_norm).toBe(false)
    expect(imported.dora_wd).toBe(true)
    expect(imported.rank_dropout).toBe(0.5)
    expect(imported.dylora_unit).toBe(8)
    expect(imported.module_dropout).toBe(0.1)
    expect(imported).not.toHaveProperty("factor")
    // Invalid scalars ("dropout=undefined") must not become form values.
    expect(imported).not.toHaveProperty("dropout")
  })

  it("repairs hidden companions when the imported lora_type is unknown", () => {
    const defaults = createDefaultModel(schema)
    const model: FormModel = {
      ...cloneFormModel(defaults),
      lora_type: "boft",
      network_module: "foreign.module",
      lokr_factor: 99,
    }
    applyReadonlyDefaults(schema, model, defaults)

    expect(model.network_module).toBe("networks.lora_anima")
    expect(model.lokr_factor).toBe(-1)
  })

  it("keeps every lokr value when the exported config is imported back", () => {
    const model = importIntoForm(exportedConfig(lokrModel()))

    expect(model.lora_type).toBe("lokr")
    expect(model.network_module).toBe("lycoris.kohya")
    expect(model.lokr_factor).toBe(16)
    expect(model.full_matrix).toBe(true)
    expect(model.use_cp).toBe(true)
    expect(model.conv_dim).toBe(8)
  })

  it("keeps lokr values when an autosaved draft is restored", () => {
    const defaults = createDefaultModel(schema)
    // The autosave stores the raw form model (TrainingPage line 435).
    const saved = cloneFormModel(lokrModel())

    const model = normalizeModelForSchema(schema, { ...cloneFormModel(defaults), ...saved })
    applyReadonlyDefaults(schema, model, defaults)

    expect(model.lora_type).toBe("lokr")
    expect(model.network_module).toBe("lycoris.kohya")
    expect(model.lokr_factor).toBe(16)
    expect(model.full_matrix).toBe(true)
  })

  it("still forces the LoRA branch consts on a plain LoRA model", () => {
    const defaults = createDefaultModel(schema)
    const model: FormModel = { ...cloneFormModel(defaults), model_train_type: "anima-lora" }
    applyReadonlyDefaults(schema, model, defaults)

    expect(model.network_module).toBe("networks.lora_anima")
    expect(model.lora_type).toBe("lora")
  })
})
