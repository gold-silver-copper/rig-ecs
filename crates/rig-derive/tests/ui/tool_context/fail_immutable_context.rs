use rig_derive::rig_tool;

struct ToolContext;

#[rig_tool]
fn immutable_context(
    #[rig(context)] context: &ToolContext,
) -> Result<(), rig_core::tool::ToolExecutionError> {
    let _ = context;
    Ok(())
}

fn main() {}
