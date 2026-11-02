import unittest
from unittest.mock import AsyncMock, patch

from agent_vla_async.cerebellum import main


class CerebellumCliTests(unittest.TestCase):
    def test_sonic_endpoint_is_independent_from_real_g1_endpoint(self):
        sonic_client = object()
        with (
            patch("agent_vla_async.cerebellum.SonicPolicyClient", return_value=sonic_client) as sonic,
            patch("agent_vla_async.cerebellum.CerebellumRuntime") as runtime_class,
        ):
            runtime_class.return_value.run = AsyncMock()
            rc = main(
                [
                    "--policy",
                    "mock",
                    "--no-arm-actions",
                    "--groot-host",
                    "real-g1-tunnel",
                    "--groot-port",
                    "5550",
                    "--sonic-policy-host",
                    "sonic-tunnel",
                    "--sonic-policy-port",
                    "5551",
                ]
            )

        self.assertEqual(0, rc)
        sonic.assert_called_once_with(
            host="sonic-tunnel",
            port=5551,
            timeout_ms=15000,
        )
        runtime_kwargs = runtime_class.call_args.kwargs
        self.assertIs(runtime_kwargs["sonic_policy"], sonic_client)
        self.assertEqual("mock-vla", runtime_kwargs["policy"].name)


if __name__ == "__main__":
    unittest.main()
