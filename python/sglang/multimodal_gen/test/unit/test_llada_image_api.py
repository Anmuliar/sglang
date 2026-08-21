# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    edits,
    generations,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
)


class TestLLaDAImageAPI(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server_args = SimpleNamespace(
            pipeline_config=LLaDAImagePipelineConfig(),
            sp_degree=2,
            input_save_path=None,
            output_path=None,
        )
        self.raw_request = SimpleNamespace(headers={})

    async def test_generation_sp_rejects_multiple_outputs_before_scheduling(self):
        request = ImageGenerationsRequest(prompt="a red car", n=2)
        fallback = HTTPException(status_code=418, detail="request was not validated")

        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.get_global_server_args",
                return_value=self.server_args,
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.build_sampling_params",
                side_effect=fallback,
            ),
            self.assertRaises(HTTPException) as context,
        ):
            await generations(request, self.raw_request)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("n=1", context.exception.detail)

    async def test_edit_sp_rejects_multiple_outputs_before_saving_input(self):
        fallback = HTTPException(status_code=418, detail="request was not validated")

        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.get_global_server_args",
                return_value=self.server_args,
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.save_image_to_path",
                new=AsyncMock(side_effect=fallback),
            ),
            self.assertRaises(HTTPException) as context,
        ):
            await edits(
                raw_request=self.raw_request,
                image=None,
                image_array=None,
                url=["https://example.com/source.png"],
                url_array=None,
                prompt="make the car blue",
                n=2,
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("n=1", context.exception.detail)


if __name__ == "__main__":
    unittest.main()
