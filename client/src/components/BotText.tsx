import * as React from 'react';
import { useRTVIClientEvent } from '@pipecat-ai/client-react';
import { RTVIEvent } from '@pipecat-ai/client-js';
import { BotLLMTextData } from '@pipecat-ai/client-js';
import { useCallback } from 'react';
import './BotText.css';

export const BotText: React.FC = () => {
  const [botText, setBotText] = React.useState('');

  useRTVIClientEvent(
    RTVIEvent.BotLlmText,
    useCallback(
      (data: BotLLMTextData) => {
        setBotText(prevText => prevText + data.text);
      },
      [setBotText]
    )
  );

  useRTVIClientEvent(
    RTVIEvent.BotStoppedSpeaking,
    useCallback(
      () => {
        setBotText(prevText => prevText + '\n\n');
      },
      [setBotText]
    )
  );

  return <pre className="bot-text">{botText}</pre>;
};
