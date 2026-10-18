import Lake
open Lake DSL

package «hivemind_spec» where
  version := v!"2.1.0"

@[default_target]
lean_lib «HiveMindSpec» where
  roots := #[`HiveMindSpec]
