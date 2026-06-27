<?php

/**
 * @file
 * Anthropic Claude adapter for the AI module.
 *
 * This adapter implements AIProviderClient using Anthropic's REST API,
 * translating shared chat-style method calls to Claude's native API.
 *
 * No external dependencies - uses Backdrop's built-in HTTP functions.
 *
 * @see https://docs.anthropic.com/en/api/getting-started
 */

class AIAnthropicAdapter extends AIAdapterBase {

  /** @var string */
  protected $baseUrl = 'https://api.anthropic.com/v1';

  public function __construct($api_key, ?AIApi $api = NULL) {
    parent::__construct($api_key, $api);
  }

  /**
   * {@inheritdoc}
   */
  protected function getDefaultHeaders(): array {
    return [
      'x-api-key'         => $this->apiKey,
      'anthropic-version' => '2023-06-01',
    ];
  }

  /**
   * Whether a model rejects the sampling parameters.
   *
   * Fable 5 and Opus 4.7/4.8 removed temperature/top_p/top_k — sending any of
   * them returns a 400. Sonnet 4.6, Opus 4.6 and earlier still accept them.
   */
  protected function modelRejectsSamplingParams(string $model): bool {
    return (bool) preg_match('/(fable|opus-4-[78])/i', $model);
  }

  /**
   * Pull system-role messages out of a shared message array.
   *
   * Anthropic takes the system prompt as a top-level `system` field, not as a
   * system-role message. convertMessages() drops system messages, so callers
   * must collect them here and set the `system` param — otherwise the persona,
   * guardrails, and site context are silently lost.
   */
  protected function extractSystemPrompt(array $messages): string {
    $system = '';
    foreach ($messages as $msg) {
      if (($msg['role'] ?? '') === 'system' && is_string($msg['content'] ?? NULL)) {
        $system .= $msg['content'] . "\n";
      }
    }
    return trim($system);
  }

  /**
   * {@inheritdoc}
   */
  public function getModels(): array {
    try {
      $url = $this->baseUrl . '/models';
      $options = [
        'method' => 'GET',
        'headers' => [
          'Accept' => 'application/json',
          'x-api-key' => $this->apiKey,
          'anthropic-version' => '2023-06-01',
        ],
        'timeout' => 10,
      ];

      $response = backdrop_http_request($url, $options);

      if (isset($response->code) && (int) $response->code === 200) {
        $data = json_decode($response->data, TRUE);
        if (!empty($data['data']) && is_array($data['data'])) {
          $models = [];
          foreach ($data['data'] as $item) {
            $id = $item['id'] ?? ($item['model'] ?? NULL);
            if (empty($id)) {
              continue;
            }

            if (!preg_match('/^claude/i', $id)) {
              continue;
            }

            $label = $item['display_name'] ?? $item['name'] ?? $id;
            if ($label === $id) {
              $models[$id] = $id;
            } else {
              $models[$id] = $label;
            }
          }

          if (!empty($models)) {
            asort($models);
            return $models;
          }
        }
      }
    }
    catch (\Exception $e) {
      watchdog('ai_provider_anthropic', 'Failed to fetch Anthropic models: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
    }

    return [];
  }

  /**
   * {@inheritdoc}
   */
  public function completions(string $model, string $prompt, $temperature, $max_tokens = 512, bool $stream_response = FALSE) {
    try {
      // Allow other modules to alter the prompt before sending (e.g., inject site context).
      if (function_exists('backdrop_alter')) {
        $context = [
          'operation' => 'completion',
          'model' => $model,
          'provider' => 'anthropic',
        ];
        backdrop_alter('ai_prompt', $prompt, $context);
      }

      $params = [
        'model' => $model,
        'max_tokens' => (int) $max_tokens ?: 512,
        'messages' => [
          [
            'role' => 'user',
            'content' => trim($prompt),
          ],
        ],
      ];
      if (!$this->modelRejectsSamplingParams($model)) {
        $params['temperature'] = max(0.0, min(1.0, (float) $temperature));
      }

      if ($stream_response) {
        return $this->handleStreamingResponse($params);
      }

      $response = $this->makeApiRequest('/messages', $params);

      // Extract text from Anthropic response.
      $text = '';
      if (isset($response['content']) && is_array($response['content'])) {
        foreach ($response['content'] as $block) {
          if (isset($block['text'])) {
            $text .= $block['text'];
          }
        }
      }

      return trim($text);
    }
    catch (\Exception $e) {
      watchdog('ai_provider_anthropic', 'Completions error: @error',
        ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * Get models by their capability.
   */
  public function getModelsByCapability($capability): array {
    $models = $this->getModels();
    if ($capability === 'text') {
      $filtered = $models;
    }
    elseif ($capability === 'vision') {
      $filtered = [];
      foreach ($models as $id => $label) {
        // Claude 3 and later generally support vision.
        if (preg_match('/claude-[3-9]/i', $id)) {
          $filtered[$id] = $label;
        }
      }
    }
    else {
      $filtered = [];
    }
    backdrop_alter('ai_model_capabilities', $filtered, $capability, $this);
    return $filtered;
  }

  /**
   * {@inheritdoc}
   */
  public function getChatModels(): array {
    return $this->getModelsByCapability('text');
  }

  /**
   * {@inheritdoc}
   */
  public function getImageModels(): array {
    return [];
  }

  /**
   * {@inheritdoc}
   */
  public function getVisionModels(): array {
    return $this->getModelsByCapability('vision');
  }

  /**
   * {@inheritdoc}
   */
  public function getEmbeddingModels(): array {
    return [];
  }

  /**
   * {@inheritdoc}
   */
  public function getModerationModels(): array {
    return [];
  }

  /**
   * {@inheritdoc}
   */
  public function chat(string $model, array $messages, $temperature, $max_tokens = 1024, bool $stream_response = FALSE, array $context_extra = []) {
    try {
      // Allow other modules to alter chat messages before sending (e.g., inject site context).
      if (function_exists('backdrop_alter') && empty($context_extra['skip_ai_message_alter'])) {
        $context = [
          'operation' => 'chat',
          'model' => $model,
          'provider' => 'anthropic',
        ];
        backdrop_alter('ai_chat_messages', $messages, $context);
      }

      // Convert messages to Anthropic format.
      $anthropic_messages = $this->convertMessages($messages);

      $params = [
        'model' => $model,
        'max_tokens' => (int) $max_tokens ?: 1024,
        'messages' => $anthropic_messages,
      ];
      // convertMessages() drops system-role messages; re-inject them as the
      // top-level system field so the persona/guardrails/site context survive.
      $system = $this->extractSystemPrompt($messages);
      if ($system !== '') {
        $params['system'] = $system;
      }
      if (!$this->modelRejectsSamplingParams($model)) {
        $params['temperature'] = max(0.0, min(1.0, (float) $temperature));
      }

      if ($stream_response) {
        return $this->handleStreamingResponse($params);
      }

      $response = $this->makeApiRequest('/messages', $params);

      // Extract text from Anthropic response.
      $text = '';
      if (isset($response['content']) && is_array($response['content'])) {
        foreach ($response['content'] as $block) {
          if (isset($block['text'])) {
            $text .= $block['text'];
          }
        }
      }

      return trim($text);
    }
    catch (\Exception $e) {
      watchdog('ai_provider_anthropic', 'Chat error: @error',
        ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * {@inheritdoc}
   *
   * Anthropic does not support image generation natively.
   */
  public function images(string $model, string $prompt, string $size, string $response_format, string $quality = 'standard', string $style = 'natural', ?string $output_format = NULL) {
    watchdog('ai_provider_anthropic',
      'Image generation is not supported by Anthropic Claude. Try a provider with image generation support, such as OpenRouter or another image-capable provider.',
      [], WATCHDOG_WARNING);
    throw new \RuntimeException('Image generation is not supported by Anthropic Claude.');
  }

  /**
   * {@inheritdoc}
   *
   * Anthropic does not support text-to-speech natively.
   */
  public function textToSpeech(string $model, string $input, string $voice, string $response_format) {
    watchdog('ai_provider_anthropic',
      'Text-to-speech is not supported by Anthropic Claude.',
      [], WATCHDOG_WARNING);
    throw new \RuntimeException('Text-to-speech is not supported by Anthropic Claude.');
  }

  /**
   * {@inheritdoc}
   *
   * Anthropic does not support speech-to-text natively.
   */
  public function speechToText(string $model, string $file, string $task = 'transcribe', $temperature = 0.4, string $response_format = 'verbose_json') {
    watchdog('ai_provider_anthropic',
      'Speech-to-text is not supported by Anthropic Claude.',
      [], WATCHDOG_WARNING);
    throw new \RuntimeException('Speech-to-text is not supported by Anthropic Claude.');
  }

  /**
   * {@inheritdoc}
   *
   * Anthropic does not provide a moderation API endpoint.
   * Claude has built-in safety policies but no content moderation score.
   */
  public function moderation(string $input, string $model = 'claude-moderation'): array {
    watchdog('ai_provider_anthropic',
      'Moderation API is not supported by Anthropic Claude. Claude has built-in safety policies.',
      [], WATCHDOG_WARNING);
    throw new \RuntimeException('Moderation is not supported by Anthropic Claude.');
  }

  /**
   * {@inheritdoc}
   *
   * Adapter must implement AIProviderClient::embedding even if the provider
   * does not support embeddings. Return an empty array and log a warning so
   * callers receive a predictable shape and administrators can diagnose.
   */
  public function embedding(string $input, string $model, bool $log = TRUE): array {
    // Record a log in ai_log if possible to show that it was attempted.
    if (isset($this->api) && method_exists($this->api, 'recordLog')) {
      $this->api->recordLog('embedding', $model, ['input' => $input], NULL, FALSE, 0, 'Anthropic does not support embeddings.', !$log);
    }
    if ($log) {
      watchdog('ai_provider_anthropic', 'Embedding requested but Anthropic does not support embeddings.', [], WATCHDOG_DEBUG);
    }
    throw new \RuntimeException('Embeddings are not supported by Anthropic Claude.');
  }

  /**
   * Handle streaming responses for Anthropic.
   *
   * @param array $params
   *   Parameters for the API request.
   *
   * @return AIStreamingResponse
   *   A streaming response object from AIAdapterBase::buildStreamingResponse().
   */
  protected function handleStreamingResponse(array $params) {
    try {
      $params['stream'] = TRUE;
      $url     = $this->baseUrl . '/messages';
      $options = $this->buildRequestOptions($params, [], 'POST', 300);

      return new AIStreamingResponse($url, $options, function ($data) {
        if (($data['delta']['type'] ?? '') === 'text_delta') {
          return $data['delta']['text'] ?? '';
        }
        return NULL;
      });
    }
    catch (\Exception $e) {
      watchdog('ai_provider_anthropic', 'Streaming error: @error',
        ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  protected function makeApiRequest($endpoint, array $body) {
    // Only /messages generation calls go through here (model listing has its
    // own fast-fail request); long generations need more than the 30s
    // makeRequest() default.
    return $this->makeRequest($this->baseUrl . $endpoint, $body, [], 'POST', 300);
  }

  /**
   * {@inheritdoc}
   */
  public function chatWithTools(string $model, array $messages, array $tools, $temperature, $max_tokens = 1024, string $tool_choice = 'auto', array $context_extra = []): array {
    try {
      $system = '';
      $filtered = [];
      foreach ($messages as $msg) {
        if (($msg['role'] ?? '') === 'system') {
          $system .= (is_string($msg['content']) ? $msg['content'] : '') . "\n";
        }
        else {
          $filtered[] = $msg;
        }
      }

      $anthropic_tools = $this->convertToolsToAnthropic($tools);
      $tc_type = ($tool_choice === 'none') ? 'none' : (($tool_choice === 'required') ? 'any' : 'auto');

      $params = [
        'model'      => $model,
        'max_tokens' => (int) $max_tokens ?: 1024,
        'messages'   => $this->convertMessages($filtered),
        'tools'      => $anthropic_tools,
        'tool_choice' => ['type' => $tc_type],
      ];
      if (!empty(trim($system))) {
        $params['system'] = trim($system);
      }

      $response = $this->makeApiRequest('/messages', $params);
      return $this->normalizeAnthropicToolResponse($response);
    }
    catch (\Exception $e) {
      watchdog('ai_provider_anthropic', 'chatWithTools error: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  protected function convertToolsToAnthropic(array $tools): array {
    $result = [];
    foreach ($tools as $tool) {
      if (($tool['type'] ?? '') === 'function' && !empty($tool['function'])) {
        $fn = $tool['function'];
        $result[] = [
          'name'         => $fn['name'],
          'description'  => $fn['description'] ?? '',
          'input_schema' => $fn['parameters'] ?? ['type' => 'object', 'properties' => []],
        ];
      }
    }
    return $result;
  }

  protected function normalizeAnthropicToolResponse(array $result): array {
    $stop_reason = $result['stop_reason'] ?? 'end_turn';
    $finish_reason = ($stop_reason === 'tool_use') ? 'tool_calls' : 'stop';
    $content = '';
    $tool_calls = [];
    foreach ($result['content'] ?? [] as $block) {
      if (($block['type'] ?? '') === 'text') {
        $content .= $block['text'] ?? '';
      }
      elseif (($block['type'] ?? '') === 'tool_use') {
        $tool_calls[] = [
          'id'        => $block['id'] ?? '',
          'name'      => $block['name'] ?? '',
          'arguments' => $block['input'] ?? [],
        ];
      }
    }
    return [
      'finish_reason' => $finish_reason,
      'content'       => trim($content),
      'tool_calls'    => $tool_calls,
      'raw'           => $result,
    ];
  }

  /**
   * Convert shared chat-style messages to Anthropic format.
   *
   * @param array $messages
   *   Array of messages in the shared chat format.
   *
   * @return array
   *   Converted messages for Anthropic API.
   */
  protected function convertMessages(array $messages): array {
    $anthropic_messages = [];

    foreach ($messages as $msg) {
      $role = $msg['role'] ?? 'user';

      // Anthropic only supports 'user' and 'assistant' roles.
      if ($role === 'system') {
        // System messages are handled via 'system' parameter in API, skip here.
        continue;
      }

      // OpenAI-shape tool results become tool_result blocks in a user turn.
      // Consecutive results merge into one turn: Anthropic requires roles to
      // alternate, and all results for one assistant turn belong together.
      if ($role === 'tool') {
        $content = $msg['content'] ?? '';
        if (!is_string($content)) {
          $content = json_encode($content);
        }
        $block = [
          'type' => 'tool_result',
          'tool_use_id' => (string) ($msg['tool_call_id'] ?? ''),
          'content' => $content,
        ];
        $last = count($anthropic_messages) - 1;
        if ($last >= 0
          && $anthropic_messages[$last]['role'] === 'user'
          && is_array($anthropic_messages[$last]['content'])
          && (($anthropic_messages[$last]['content'][0]['type'] ?? '') === 'tool_result')) {
          $anthropic_messages[$last]['content'][] = $block;
        }
        else {
          $anthropic_messages[] = ['role' => 'user', 'content' => [$block]];
        }
        continue;
      }

      // Convert the shared 'assistant' role to 'assistant' (Anthropic)
      $anthropic_role = ($role === 'assistant') ? 'assistant' : 'user';
      $content = $this->convertContentAnthropic($msg['content'] ?? '');

      // OpenAI-shape assistant tool_calls become tool_use blocks. Without
      // this the calls were dropped and a text-less assistant turn became
      // empty content, which Anthropic rejects with a 400.
      if ($anthropic_role === 'assistant' && !empty($msg['tool_calls']) && is_array($msg['tool_calls'])) {
        $blocks = [];
        if (is_string($content)) {
          if (trim($content) !== '') {
            $blocks[] = ['type' => 'text', 'text' => $content];
          }
        }
        elseif (is_array($content)) {
          $blocks = $content;
        }
        foreach ($msg['tool_calls'] as $tc) {
          $args = $tc['function']['arguments'] ?? ($tc['arguments'] ?? []);
          if (is_string($args)) {
            $args = json_decode($args, TRUE);
          }
          // 'input' must serialize as a JSON object — an empty PHP array
          // would encode as [] and be rejected.
          $blocks[] = [
            'type' => 'tool_use',
            'id' => (string) ($tc['id'] ?? ''),
            'name' => (string) ($tc['function']['name'] ?? ($tc['name'] ?? '')),
            'input' => is_array($args) && $args !== [] ? $args : new \stdClass(),
          ];
        }
        $anthropic_messages[] = ['role' => 'assistant', 'content' => $blocks];
        continue;
      }

      // Anthropic rejects empty message content; skip empty turns.
      if ((is_string($content) && trim($content) === '') || (is_array($content) && $content === [])) {
        continue;
      }

      $anthropic_messages[] = [
        'role' => $anthropic_role,
        'content' => $content,
      ];
    }

    return $anthropic_messages;
  }

  /**
   * Convert shared chat message content to Anthropic content blocks.
   */
  protected function convertContentAnthropic($content) {
    if (!is_array($content)) {
      return (string) $content;
    }

    $blocks = [];
    foreach ($content as $block) {
      if (is_string($block)) {
        $blocks[] = ['type' => 'text', 'text' => $block];
        continue;
      }
      if (!is_array($block)) {
        $blocks[] = ['type' => 'text', 'text' => json_encode($block)];
        continue;
      }

      $type = $block['type'] ?? '';
      if ($type === 'text') {
        $blocks[] = ['type' => 'text', 'text' => (string) ($block['text'] ?? '')];
        continue;
      }

      if ($type === 'image_url') {
        $image_url = $block['image_url']['url'] ?? '';
        if (preg_match('/^data:([^;]+);base64,(.+)$/', $image_url, $matches)) {
          $blocks[] = [
            'type' => 'image',
            'source' => [
              'type' => 'base64',
              'media_type' => $matches[1],
              'data' => $matches[2],
            ],
          ];
        }
      }
    }

    return $blocks;
  }
}
