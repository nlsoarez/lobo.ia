"""
Modulo de analise de noticias para melhorar decisoes de trading.
Busca noticias do Yahoo Finance e realiza analise de sentimento.
"""

import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re

from system_logger import system_logger


class NewsAnalyzer:
    """
    Analisador de noticias para criptomoedas e acoes.
    Usa palavras-chave para determinar sentimento (positivo/negativo/neutro).
    """

    # Palavras positivas em ingles e portugues
    POSITIVE_WORDS = [
        # English
        'surge', 'soar', 'rally', 'bull', 'bullish', 'gain', 'gains', 'rise', 'rises',
        'rising', 'up', 'high', 'higher', 'growth', 'grow', 'growing', 'profit',
        'profits', 'positive', 'boost', 'jump', 'jumps', 'record', 'success',
        'breakthrough', 'upgrade', 'buy', 'outperform', 'beat', 'beats', 'strong',
        'strength', 'recover', 'recovery', 'boom', 'adoption', 'approved', 'launch',
        'partnership', 'invest', 'investment', 'institutional', 'etf', 'halving',
        # Portugues
        'alta', 'sobe', 'subindo', 'valoriza', 'valorizacao', 'lucro', 'lucros',
        'ganho', 'ganhos', 'positivo', 'recorde', 'sucesso', 'crescimento',
        'recuperacao', 'parceria', 'aprovado', 'investimento'
    ]

    # Palavras negativas em ingles e portugues
    NEGATIVE_WORDS = [
        # English
        'crash', 'plunge', 'drop', 'drops', 'fall', 'falls', 'falling', 'decline',
        'declining', 'down', 'low', 'lower', 'loss', 'losses', 'negative', 'bear',
        'bearish', 'sell', 'selloff', 'dump', 'fear', 'panic', 'risk', 'risky',
        'warning', 'warn', 'crisis', 'trouble', 'problem', 'hack', 'hacked',
        'scam', 'fraud', 'ban', 'banned', 'lawsuit', 'sue', 'investigation',
        'regulation', 'restrict', 'fine', 'penalty', 'bankruptcy', 'collapse',
        'liquidation', 'recession', 'inflation', 'correction',
        # Portugues
        'queda', 'cai', 'caindo', 'desvaloriza', 'desvalorizacao', 'perda',
        'perdas', 'negativo', 'risco', 'crise', 'fraude', 'golpe', 'proibido',
        'investigacao', 'multa', 'falencia', 'recessao', 'inflacao'
    ]

    def __init__(self):
        """Inicializa o analisador de noticias."""
        self.cache: Dict[str, Dict] = {}
        self.cache_duration = timedelta(minutes=15)

    def get_news(self, symbol: str, max_news: int = 10) -> List[Dict]:
        """
        Busca noticias para um simbolo.

        Args:
            symbol: Simbolo do ativo (ex: BTC-USD, PETR4.SA)
            max_news: Numero maximo de noticias

        Returns:
            Lista de noticias com titulo, link e data
        """
        try:
            # Verifica cache
            cache_key = f"{symbol}_{max_news}"
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                if datetime.now() - cached['timestamp'] < self.cache_duration:
                    return cached['news']

            # Busca noticias via yfinance
            ticker = yf.Ticker(symbol)
            news_data = ticker.news

            if not news_data:
                return []

            news_list = []
            for item in news_data[:max_news]:
                news_item = {
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'published': datetime.fromtimestamp(
                        item.get('providerPublishTime', 0)
                    ).strftime('%Y-%m-%d %H:%M') if item.get('providerPublishTime') else '',
                    'type': item.get('type', 'STORY')
                }
                news_list.append(news_item)

            # Atualiza cache
            self.cache[cache_key] = {
                'timestamp': datetime.now(),
                'news': news_list
            }

            return news_list

        except Exception as e:
            system_logger.debug(f"Erro ao buscar noticias para {symbol}: {e}")
            return []

    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analisa sentimento de um texto.

        Args:
            text: Texto para analisar (titulo da noticia)

        Returns:
            Tupla (sentimento, score) onde sentimento eh 'positive', 'negative' ou 'neutral'
            e score eh um valor entre -1.0 e 1.0
        """
        text_lower = text.lower()

        positive_count = 0
        negative_count = 0

        # Conta palavras positivas
        for word in self.POSITIVE_WORDS:
            if re.search(r'\b' + word + r'\b', text_lower):
                positive_count += 1

        # Conta palavras negativas
        for word in self.NEGATIVE_WORDS:
            if re.search(r'\b' + word + r'\b', text_lower):
                negative_count += 1

        total = positive_count + negative_count

        if total == 0:
            return 'neutral', 0.0

        # Calcula score (-1.0 a 1.0)
        score = (positive_count - negative_count) / total

        if score > 0.2:
            sentiment = 'positive'
        elif score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return sentiment, score

    def get_news_sentiment(self, symbol: str, max_news: int = 10) -> Dict:
        """
        Busca noticias e analisa sentimento geral.

        Args:
            symbol: Simbolo do ativo
            max_news: Numero maximo de noticias

        Returns:
            Dicionario com noticias analisadas e sentimento geral
        """
        news = self.get_news(symbol, max_news)

        if not news:
            return {
                'symbol': symbol,
                'news_count': 0,
                'overall_sentiment': 'neutral',
                'overall_score': 0.0,
                'news': [],
                'recommendation': 'HOLD'
            }

        analyzed_news = []
        total_score = 0.0

        for item in news:
            sentiment, score = self.analyze_sentiment(item['title'])
            analyzed_news.append({
                **item,
                'sentiment': sentiment,
                'score': score
            })
            total_score += score

        # Calcula media
        avg_score = total_score / len(news) if news else 0.0

        if avg_score > 0.2:
            overall_sentiment = 'positive'
            recommendation = 'BUY'
        elif avg_score < -0.2:
            overall_sentiment = 'negative'
            recommendation = 'SELL'
        else:
            overall_sentiment = 'neutral'
            recommendation = 'HOLD'

        result = {
            'symbol': symbol,
            'news_count': len(news),
            'overall_sentiment': overall_sentiment,
            'overall_score': avg_score,
            'news': analyzed_news,
            'recommendation': recommendation,
            'analyzed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        system_logger.info(
            f"📰 {symbol}: {len(news)} noticias | "
            f"Sentimento: {overall_sentiment.upper()} ({avg_score:+.2f}) | "
            f"Recomendacao: {recommendation}"
        )

        return result

    def get_market_sentiment(self, symbols: List[str]) -> Dict:
        """
        Analisa sentimento de multiplos ativos.

        Args:
            symbols: Lista de simbolos

        Returns:
            Dicionario com sentimento de cada ativo e geral do mercado
        """
        results = {}
        total_score = 0.0
        valid_count = 0

        for symbol in symbols:
            result = self.get_news_sentiment(symbol)
            results[symbol] = result

            if result['news_count'] > 0:
                total_score += result['overall_score']
                valid_count += 1

        # Calcula sentimento geral do mercado
        if valid_count > 0:
            market_score = total_score / valid_count
            if market_score > 0.15:
                market_sentiment = 'bullish'
            elif market_score < -0.15:
                market_sentiment = 'bearish'
            else:
                market_sentiment = 'neutral'
        else:
            market_score = 0.0
            market_sentiment = 'neutral'

        return {
            'assets': results,
            'market_sentiment': market_sentiment,
            'market_score': market_score,
            'total_assets_analyzed': valid_count,
            'analyzed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def should_boost_signal(self, symbol: str, current_signal: str) -> Tuple[bool, float]:
        """
        Verifica se notícias devem reforçar ou enfraquecer um sinal.

        Args:
            symbol: Simbolo do ativo
            current_signal: Sinal atual ('BUY', 'SELL', 'HOLD')

        Returns:
            Tupla (deve_modificar, fator_ajuste)
            fator_ajuste: 0.8 a 1.2 (reduz ou aumenta forca do sinal)
        """
        try:
            sentiment = self.get_news_sentiment(symbol, max_news=5)

            if sentiment['news_count'] == 0:
                return False, 1.0

            news_score = sentiment['overall_score']
            news_rec = sentiment['recommendation']

            # Se sinal e noticias concordam, aumenta forca
            if current_signal == 'BUY' and news_rec == 'BUY':
                return True, 1.15  # +15% na forca

            if current_signal == 'SELL' and news_rec == 'SELL':
                return True, 1.15  # +15% na forca

            # Se sinal e noticias discordam, reduz forca
            if current_signal == 'BUY' and news_rec == 'SELL':
                return True, 0.85  # -15% na forca

            if current_signal == 'SELL' and news_rec == 'BUY':
                return True, 0.85  # -15% na forca

            # Noticias neutras ou HOLD nao modificam
            return False, 1.0

        except Exception as e:
            system_logger.debug(f"Erro ao verificar noticias para {symbol}: {e}")
            return False, 1.0

    def print_news_report(self, symbols: List[str]):
        """
        Imprime relatorio de noticias formatado.

        Args:
            symbols: Lista de simbolos
        """
        print("\n" + "=" * 70)
        print("RELATORIO DE NOTICIAS E SENTIMENTO DO MERCADO")
        print("=" * 70)

        for symbol in symbols:
            result = self.get_news_sentiment(symbol)

            print(f"\n{'='*50}")
            print(f"  {symbol}")
            print(f"{'='*50}")
            print(f"  Noticias: {result['news_count']}")
            print(f"  Sentimento: {result['overall_sentiment'].upper()}")
            print(f"  Score: {result['overall_score']:+.2f}")
            print(f"  Recomendacao: {result['recommendation']}")

            if result['news']:
                print(f"\n  Ultimas noticias:")
                for i, news in enumerate(result['news'][:3], 1):
                    sentiment_icon = {
                        'positive': '🟢',
                        'negative': '🔴',
                        'neutral': '⚪'
                    }.get(news['sentiment'], '⚪')

                    title = news['title'][:60] + '...' if len(news['title']) > 60 else news['title']
                    print(f"    {i}. {sentiment_icon} {title}")

        print("\n" + "=" * 70)


# Funcao auxiliar para uso rapido
def analyze_news(symbol: str) -> Dict:
    """
    Funcao auxiliar para analisar noticias de um simbolo.

    Args:
        symbol: Simbolo do ativo

    Returns:
        Resultado da analise
    """
    analyzer = NewsAnalyzer()
    return analyzer.get_news_sentiment(symbol)


if __name__ == "__main__":
    # Teste do modulo
    analyzer = NewsAnalyzer()

    # Testa com algumas criptos
    test_symbols = ['BTC-USD', 'ETH-USD', 'PETR4.SA']
    analyzer.print_news_report(test_symbols)
