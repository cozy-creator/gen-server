import json
import unittest

from gen_server.api import get_react_components, get_checkpoints, handle_post
from gen_server.globals import CHECKPOINT_FILES


class GenServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_get_react_components(self):
        res = await get_react_components({})
        newJ = json.loads(res.text)
        keys = newJ.keys().__len__()
        self.assertEqual(keys, 1)

    async def test_get_checkpoints(self):
        res = await get_checkpoints({})
        self.assertEqual(res.text, '{}')


if __name__ == '__main__':
    unittest.main()