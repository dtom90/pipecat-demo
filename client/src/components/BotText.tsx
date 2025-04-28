import * as React from 'react';
import { useRTVIClientEvent } from '@pipecat-ai/client-react';
import { RTVIEvent } from '@pipecat-ai/client-js';
import { BotLLMTextData } from '@pipecat-ai/client-js';
import { useCallback, useRef, useEffect } from 'react';
import './BotText.css';

export const BotText: React.FC = () => {
  const [botText, setBotText] = React.useState('');
  const textContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll effect when text changes
  useEffect(() => {
    if (textContainerRef.current) {
      textContainerRef.current.scrollTop = textContainerRef.current.scrollHeight;
    }
  }, [botText]);

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

  return <div className="bot-text-container" ref={textContainerRef}>
    <pre className="bot-text">{botText}</pre>
  </div>;
};
