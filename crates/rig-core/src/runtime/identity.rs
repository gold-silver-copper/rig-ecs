//! Stable domain and tenant identities.
//!
//! Bevy [`Entity`](bevy_ecs::entity::Entity) values remain runtime-local. These
//! identifiers are the only identities permitted to cross persistence,
//! process, or diagnostic boundaries.

use bevy_ecs::prelude::Component;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Persistent identity for a domain entity.
#[derive(Component, Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct StableId(String);

impl StableId {
    pub(crate) fn generated(value: String) -> Self {
        debug_assert!(!value.is_empty());
        Self(value)
    }

    /// Creates a stable identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, IdentityError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(IdentityError::Empty);
        }
        Ok(Self(value))
    }

    /// Returns the serialized identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Tenant scope participating in runtime authorization decisions.
#[derive(Component, Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct TenantId(String);

impl TenantId {
    /// Creates a tenant scope.
    pub fn new(value: impl Into<String>) -> Result<Self, IdentityError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(IdentityError::Empty);
        }
        Ok(Self(value))
    }

    /// Returns the tenant identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Invalid stable or tenant identity.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum IdentityError {
    /// Empty identifiers are not valid persistence keys.
    #[error("identity must not be empty")]
    Empty,
}
