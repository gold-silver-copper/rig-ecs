//! Cross-platform time helpers used by runtime effects.

use std::{future::Future, time::Duration};

/// Error returned by [`timeout`] when the future does not complete in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Elapsed;

impl std::fmt::Display for Elapsed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("future timed out")
    }
}

impl std::error::Error for Elapsed {}

/// Await `future`, dropping it and returning [`Elapsed`] when `duration` expires.
pub(crate) async fn timeout<F>(duration: Duration, future: F) -> Result<F::Output, Elapsed>
where
    F: Future,
{
    use futures::future::{Either, select};

    let delay = futures_timer::Delay::new(duration);
    futures::pin_mut!(future);
    futures::pin_mut!(delay);
    match select(future, delay).await {
        Either::Left((output, _)) => Ok(output),
        Either::Right(((), _)) => Err(Elapsed),
    }
}

#[cfg(test)]
mod tests {
    use super::{Elapsed, timeout};
    use std::time::Duration;

    #[tokio::test]
    async fn completes_before_deadline() {
        assert_eq!(timeout(Duration::from_secs(5), async { 42 }).await, Ok(42));
    }

    #[tokio::test]
    async fn expires_pending_future() {
        let result = timeout(Duration::from_millis(20), std::future::pending::<()>()).await;
        assert_eq!(result, Err(Elapsed));
    }

    #[tokio::test]
    async fn ready_future_wins_zero_duration() {
        assert_eq!(timeout(Duration::ZERO, async { 7 }).await, Ok(7));
    }
}
