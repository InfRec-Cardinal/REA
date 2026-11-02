import unittest

from agent_vla_async.mailbox import LatestValue


class LatestValueTest(unittest.IsolatedAsyncioTestCase):
    async def test_latest_value_overwrites_stale_work(self):
        mailbox = LatestValue()
        await mailbox.publish("old")
        version = await mailbox.publish("new")
        current_version, value = mailbox.snapshot()
        self.assertEqual(version, current_version)
        self.assertEqual("new", value)

    async def test_clear_wakes_consumer(self):
        mailbox = LatestValue()
        version = await mailbox.publish("value")
        waiter = __import__("asyncio").create_task(mailbox.wait_after(version))
        cleared = await mailbox.clear()
        next_version, value = await waiter
        self.assertEqual(cleared, next_version)
        self.assertIsNone(value)


if __name__ == "__main__":
    unittest.main()

