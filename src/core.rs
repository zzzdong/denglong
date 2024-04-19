use std::fmt::format;

use anyhow::{bail, Error as E, Result};

use candle_transformers::models::mpt::{Config, Model as M};
use candle_transformers::models::quantized_mpt::Model as Q;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
// use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone)]
enum Model {
    M(M),
    Q(Q),
}

impl Model {
    fn forward(&mut self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::M(model) => model.forward(xs),
            Self::Q(model) => model.forward(xs),
        }
    }
}

pub(crate) struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
            device: device.clone(),
        }
    }

    pub(crate) fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        use std::io::Write;
        println!("starting the inference loop");
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_vocab(true).get("<fim_middle>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        
        let mut output = String::new();


        std::io::stdout().flush()?;
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;

            output.push_str(&token);

        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        
        Ok(output)
    }
}

pub struct Core {
    model: Model,
    device: Device,
    tokenizer: Tokenizer,
}



impl Core {
    pub fn new() -> Result<Self> {
        let config = Config::replit_code_v1_5_3b();
        let device = Device::Cpu;
        let model_filename = "/home/alex/.cache/huggingface/hub/models--lmz--candle-replit-code/snapshots/97a4aa8b2b40e55d56d50a4dae5ff1ecc5f92e14/model-replit-code-v1_5-q4k.gguf";
        let tokenizer_filename = "/home/alex/.cache/huggingface/hub/models--lmz--candle-replit-code/snapshots/97a4aa8b2b40e55d56d50a4dae5ff1ecc5f92e14/tokenizer.json";

        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &model_filename,
            &device,
        ).map_err(|err|E::msg(format!("model_filename err: {err}")))?;
        let model = Model::Q(Q::new(&config, vb.pp("transformer"))?);
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|err|E::msg(format!("tokenizer_filename err: {err}")))?;

        Ok(Core {
            model,
            device,
            tokenizer,
        })
    }

    pub fn new_generation(&self) -> Result<TextGeneration> {
        let ret = TextGeneration::new(
            self.model.clone(),
            self.tokenizer.clone(),
            299792458,
            None, // TODO
            None, // TODO
            1.0,
            64,
            false,
            &self.device,
        );

        Ok(ret)
    }
}
