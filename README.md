# RAG for RIOT OS

A Retrieval-Augmented Generation (RAG) toolchain for the RIOT operating system. This repository helps you build local vector search indices over RIOT's documentation and example code to power LLM-assisted development, code generation, and explanations.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Documentation RAG](#documentation-rag)

   * [1. Clone RIOT](#1-clone-riot)
   * [2. Generate Documentation](#2-generate-documentation)
   * [3. Chunk & Embed Documentation](#3-chunk--embed-documentation)
   * [4. Query the RAG](#4-query-the-rag)
3. [Autoencoder (Optional)](#autoencoder-optional)
4. [Code RAG (Examples Directory)](#code-rag-examples-directory)
5. [Usage Examples](#example)
6. [License](#license)

---

## Prerequisites

* Python 3.8+
* [Doxygen](https://www.doxygen.nl/) to generate API documentation
* Required pip packages:

  ```bash
  pip install all required packaged please 
  ```

---

## Documentation RAG

Leverage RIOT's API docs for retrieval-augmented prompts.

### 1. Clone RIOT

```bash
git clone https://github.com/RIOT-OS/RIOT.git
cd RIOT
```

### 2. Generate Documentation

Use Doxygen to build HTML or XML docs locally:

```bash
doxygen Doxyfile
# Output will be in ./doc or ./html by default
```

### 3. Chunk & Embed Documentation

1. **Chunk** the generated docs:

   ```bash
   python3 RIOTDocuChunker2.py path/to/RIOT/doc/html
   ```

   * Produces `riot_chunks.json` containing overlapping text chunks and metadata.

2. **Embed** chunks into a vector database:

   ```bash
   python3 RIOTRRAGDocuDB3.py riot_chunks.json
   ```

   * Creates a ChromaDB at `./riot_vector_db` (default path, configurable).

### 4. Query the RAG

Retrieve relevant documentation snippets for any query:

```bash
python3 RIORDocuRAGRequest2.py "<your query>"
```

The script returns:

* Your original user query
* Top matching documentation chunks
* A ready-to-use prompt template for your LLM

---

## Autoencoder (Optional)

Compress embeddings to speed up search and potentially improve relevance.

1. **Standard Autoencoder**:

   ```bash
   python3 AutoencoderRIOT2.py
   ```
2. **Triplet Autoencoder** (margin-based grouping):

   ```bash
   python3 AutoencoderRIOTTriplet.py --epochs 100 --lambda-triplet 5.0 --margin 1.5
   ```

**Note:** Compare performance with the uncompressed RAG to evaluate impact.

To query with compressed vectors:

```bash
python3 RIORDocuRAGRequestCompressed.py "<your query>"
python3 RIORDocuRAGRequestCompressedTriplet.py "<your query>"
```

---

## Code RAG (Examples Directory)

Perform RAG over RIOT's `examples/` codebase.

1. **Set examples directory** in `chunker.py` (line 11):

   ```python
   EXAMPLES_DIR = "/path/to/RIOT/examples"
   ```
2. **Chunk** the examples:

   ```bash
   python3 chunker.py
   ```
3. **Embed** the chunks:

   ```bash
   python3 embedder.py
   ```
4. **Query** the example RAG:

   ```bash
   python3 request.py "<your query>"
   ```

> Warning: If no example matches your query, results may be irrelevant. Use alongside Documentation RAG.

---



## License

This project is licensed under the [MIT License](./LICENSE).


# Example
As an example I has the following query: "I want to make a udp server that returns a resource using gnrc", first I ran this query in RIORDocuRAGRequest2.py and I got this:
You are a RIOT OS expert assistant. Use the following documentation chunks to answer the user's question about RIOT OS.

**User Question:** I want to make a udp server that returns a resource using gnrc

**RIOT OS Documentation Context:**
The following 8 chunks were retrieved from the official RIOT OS documentation based on semantic similarity and relevance to your question:


---
**Chunk 1** (Relevance: 0.399)
**Module:** gnrc
**Category:** core
**File:** gnrc_2udp_8h.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\gnrc_2udp_8h.html
**Section:** Unknown

```
sys/include/net/gnrc/udp.h File Reference Toggle navigation Documentation The friendly Operating System for the Internet of Things Macros | Functions udp.h File ReferenceNetworking » Generic (GNRC) network stack » UDP UDP GNRC definition. More... Detailed Description UDP GNRC definition. AuthorHauke Petersen hauke.nosp@m..pet.nosp@m.ersen.nosp@m.@fu-.nosp@m.berli.nosp@m.n.de Definition in file udp.h. #include <stdint.h> #include "byteorder.h" #include "net/gnrc.h" #include "net/udp.h" Include dependency graph for udp.h: This browser is not able to show SVG: try Firefox, Chrome, Safari, or Opera instead. This graph shows which files directly or indirectly include this file: This browser is not able to show SVG: try Firefox, Chrome, Safari, or Opera instead. Go to the source code of this file. Macros #define CONFIG_GNRC_UDP_MSG_QUEUE_SIZE_EXP   (3U) Default message queue size for the UDP thread (as exponent of 2^n). More... #define GNRC_UDP_PRIO   (THREAD_PRIORITY_MAIN - 2) Priority of the UDP thread. #define GNRC_UDP_STACK_SIZE   ((THREAD_STACKSIZE_SMALL) - 64) Default stack size to use for the UDP thread. More... #define GNRC_UDP_MSG_QUEUE_SIZE   (1 << CONFIG_GNRC_UDP_MSG_QUEUE_SIZE_EXP) Message queue size to use for the UDP thread. Functions int gnrc_udp_calc_csum (gnrc_pktsnip_t *hdr, gnrc_pktsnip_t *pseudo_hdr) Calculate the checksum for the given packet. More... gnrc_pktsnip_t * gnrc_udp_hdr_build (gnrc_pktsnip_t *payload, uint16_t src, uint16_t dst) Allocate and initialize a fresh UDP header in the packet buffer. More... int gnrc_udp_init (void) Initialize and start UDP. More... Generated on Sat Jul 12 2025 00:26:07 by 1.9.1
```


---
**Chunk 2** (Relevance: 0.348)
**Module:** udp
**Category:** sys
**File:** group__net__sock__udp.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\group__net__sock__udp.html
**Section:** Unknown

```
UDP sock API Toggle navigation Documentation The friendly Operating System for the Internet of Things Files | Data Structures | Typedefs | Functions UDP sock APINetworking » Sock API Sock submodule for UDP. More... Detailed Description Sock submodule for UDP. How To Use First you need to include a module that implements this API in your application's Makefile. For example the implementation for GNRC is called gnrc_sock_udp. A Simple UDP Echo Server #include <stdio.h> #include "net/sock/udp.h" uint8_t buf[128]; int main(void) { sock_udp_ep_t local = SOCK_IPV6_EP_ANY; sock_udp_t sock; local.port = 12345; if (sock_udp_create(&sock, &local, NULL, 0) < 0) { puts("Error creating UDP sock"); return 1; } while (1) { sock_udp_ep_t remote; ssize_t res; if ((res = sock_udp_recv(&sock, buf, sizeof(buf), SOCK_NO_TIMEOUT, &remote)) >= 0) { puts("Received a message"); if (sock_udp_send(&sock, buf, res, &remote) < 0) { puts("Error sending reply"); } } } return 0; } sock_udp_createint sock_udp_create(sock_udp_t *sock, const sock_udp_ep_t *local, const sock_udp_ep_t *remote, uint16_t flags)Creates a new UDP sock object. sock_udp_sendstatic ssize_t sock_udp_send(sock_udp_t *sock, const void *data, size_t len, const sock_udp_ep_t *remote)Sends a UDP message to remote end point.Definition: udp.h:754 sock_udp_recvstatic ssize_t sock_udp_recv(sock_udp_t *sock, void *data, size_t max_len, uint32_t timeout, sock_udp_ep_t *remote)Receives a UDP message from a remote end point.Definition: udp.h:530 SOCK_IPV6_EP_ANY#define SOCK_IPV6_EP_ANYAddress to bind to any IPv6 address.Definition: sock.h:165 SOCK_NO_TIMEOUT#define SOCK_NO_TIMEOUTSpecial value meaning "wait forever" (don't timeout)Definition: sock.h:172 udp.hUDP sock definitions. _sock_tl_epCommon IP-based transport layer end point.Definition: sock.h:214 _sock_tl_ep::portuint16_t porttransport layer port (in host byte order)Definition: sock.h:246 sock_udpUDP sock type.Definition: sock_types.h:128 Above you see a simple UDP echo server. Don't forget to also include the IPv6 module of your networking implementation (e.g. gnrc_ipv6_default for Generic (GNRC) network stack GNRC) and at least one network device. After including the header file for UDP sock, we create some buffer space buf to store the data received by the server: #include "net/sock/udp.h" uint8_t buf[128]; To be able to listen for incoming packets we bind the sock by setting a local end point with a port (12345 in this case). We then proceed to create the sock. It is bound to local and thus listens for UDP packets with destination port 12345. Since we don't need any further configuration we set the flags to 0. In case of an error we stop the program: sock_udp_ep_t local = SOCK_IPV6_EP_ANY; sock_udp_t sock; local.port = 12345; if (sock_udp_create(&sock, &local, NULL, 0) < 0) { puts("Error creating UDP sock"); return 1; } The application then waits indefinitely for an incoming message in buf from remote. If we want to timeout this wait period we could alternatively set the timeout parameter of sock_udp_recv() to a value != SOCK_NO_TIMEOUT. If an error occurs on receive we just ignore it and continue looping. If we
```


---
**Chunk 3** (Relevance: 0.332)
**Module:** gnrc
**Category:** sys
**File:** group__net__gnrc__nettype__udp.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\group__net__gnrc__nettype__udp.html
**Section:** Unknown

```
gnrc_nettype_udp Toggle navigation Documentation The friendly Operating System for the Internet of Things gnrc_nettype_udpNetworking » Generic (GNRC) network stack | Generic pseudomodules » gnrc_nettype: Protocol type Enables GNRC_NETTYPE_UDP. Enables GNRC_NETTYPE_UDP. Generated on Sat Jul 12 2025 00:28:03 by 1.9.1
```


---
**Chunk 4** (Relevance: 0.316)
**Module:** gnrc
**Category:** sys
**File:** group__net__sock__async__event.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\group__net__sock__async__event.html
**Section:** Unknown

```
_udp_recv(sock_udp_t *sock, void *data, size_t max_len, uint32_t timeout, sock_udp_ep_t *remote)Receives a UDP message from a remote end point.Definition: udp.h:530 SOCK_IPV6_EP_ANY#define SOCK_IPV6_EP_ANYAddress to bind to any IPv6 address.Definition: sock.h:165 event_loopstatic void event_loop(event_queue_t *queue)Simple event loop.Definition: event.h:486 event_queue_initstatic void event_queue_init(event_queue_t *queue)Initialize an event queue.Definition: event.h:184 event.hAsynchronous sock using Event Queue definitions. udp.hUDP sock definitions. PTRTAGevent queue structureDefinition: event.h:153 _sock_tl_epCommon IP-based transport layer end point.Definition: sock.h:214 _sock_tl_ep::portuint16_t porttransport layer port (in host byte order)Definition: sock.h:246 sock_udpUDP sock type.Definition: sock_types.h:128 Above you see a simple UDP echo server using Event Queue. Don't forget to also include the IPv6 module of your networking implementation (e.g. gnrc_ipv6_default for Generic (GNRC) network stack GNRC) and at least one network device. After including the header file for UDP sock and asynchronous handling, we create the event queue queue and allocate some buffer space buf to store the data received by the server: #include "net/sock/udp.h" #include "net/sock/async/event.h" event_queue_t queue; uint8_t buf[128]; We then define an event handler in form of a function. The event handler checks if the triggering event was a receive event by checking the flags provided with sock_event_t::type. If it is a receive event it copies the incoming message to buf and its sender into remote using sock_udp_recv(). Note, that we use sock_udp_recv() non-blocking by setting timeout to 0. If an error occurs on receive, we just ignore it and return from the event handler. If we receive a message we use its remote to reply. In case of an error on send, we print an according message: void handler(sock_udp_t *sock, sock_async_flags_t type, void *arg) { (void)arg; if (type & SOCK_ASYNC_MSG_RECV) { sock_udp_ep_t remote; ssize_t res; if ((res = sock_udp_recv(sock, buf, sizeof(buf), 0, &remote)) >= 0) { puts("Received a message"); if (sock_udp_send(sock, buf, res, &remote) < 0) { puts("Error sending reply"); } } } } To be able to listen for incoming packets we bind the sock by setting a local end point with a port (12345 in this case). We then proceed to create the sock. It is bound to local and thus listens for UDP packets with destination port 12345. Since we don't need any further configuration we set the flags to 0. In case of an error we stop the program: sock_udp_ep_t local = SOCK_IPV6_EP_ANY; sock_udp_t sock; local.port = 12345; if (sock_udp_create(&sock, &local, NULL, 0) < 0) { puts("Error creating UDP sock"); return 1; } Finally, we initialize the event queue queue, initialize asynchronous event handling for sock using it and the previously defined event handler, and go into an endless loop to handle all occurring events on queue: event_queue_init(&queue); sock_udp_event_init(&sock, &queue, handler, NULL); event_loop(&queue); Files file  event.h Asynchronous sock using Event Queue definitions. file  sock_async_ctx.h Type definitions for asynchronous socks with Event Queue. Data Structures union  sock_event_cb_t Generalized callback type. More... struct  sock_event_t Event definition
```


---
**Chunk 5** (Relevance: 0.314)
**Module:** gnrc
**Category:** core
**File:** group__net__gnrc__udp.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\group__net__gnrc__udp.html
**Section:** Unknown

```
UDP Toggle navigation Documentation The friendly Operating System for the Internet of Things Modules | Files | Macros | Functions UDPNetworking » Generic (GNRC) network stack GNRC's implementation of the UDP protocol. More... Detailed Description GNRC's implementation of the UDP protocol. Modules GNRC UDP compile configurations Files file  udp.h UDP GNRC definition. Macros #define GNRC_UDP_MSG_QUEUE_SIZE   (1 << CONFIG_GNRC_UDP_MSG_QUEUE_SIZE_EXP) Message queue size to use for the UDP thread. Functions int gnrc_udp_calc_csum (gnrc_pktsnip_t *hdr, gnrc_pktsnip_t *pseudo_hdr) Calculate the checksum for the given packet. More... gnrc_pktsnip_t * gnrc_udp_hdr_build (gnrc_pktsnip_t *payload, uint16_t src, uint16_t dst) Allocate and initialize a fresh UDP header in the packet buffer. More... int gnrc_udp_init (void) Initialize and start UDP. More... Function Documentation �� gnrc_udp_calc_csum() int gnrc_udp_calc_csum ( gnrc_pktsnip_t * hdr, gnrc_pktsnip_t * pseudo_hdr ) Calculate the checksum for the given packet. Parameters [in]hdrPointer to the UDP header [in]pseudo_hdrPointer to the network layer header Returns0 on success -EBADMSG if hdr is not of type GNRC_NETTYPE_UDP -EFAULT if hdr or pseudo_hdr is NULL -ENOENT if gnrc_pktsnip_t::type of pseudo_hdr is not known �� gnrc_udp_hdr_build() gnrc_pktsnip_t* gnrc_udp_hdr_build ( gnrc_pktsnip_t * payload, uint16_t src, uint16_t dst ) Allocate and initialize a fresh UDP header in the packet buffer. Parameters [in]payloadPayload contained in the UDP packet [in]srcSource port in host byte order [in]dstDestination port in host byte order Returnspointer to the newly created (and allocated) header NULL on src == NULL, dst == NULL, src_len != 2, dst_len != 2 or on allocation error Preconditionsrc > 0 and dst > 0 �� gnrc_udp_init() int gnrc_udp_init ( void ) Initialize and start UDP. ReturnsPID of the UDP thread negative value on error Generated on Sat Jul 12 2025 00:27:55 by 1.9.1
```


---
**Chunk 6** (Relevance: 0.307)
**Module:** gnrc
**Category:** core
**File:** group__net__gnrc.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\group__net__gnrc.html
**Section:** Unknown

```
 to be sent is added to the packet buffer. This ensures its intactness during the sending process. After the data to be sent has been added to the packet buffer, its parent data structure can safely be freed or reused. Then, the pkt will be sent to all threads that registered for GNRC_NETTYPE_UDP and the demux context 80. Every registered thread will receive a GNRC_NETAPI_MSG_TYPE_SND command and can access the Packet. Note that at this point, the threads receiving pkt act as its owners, so please don't modify pkt after calling any dispatch function. If gnrc_netapi_dispatch_send() is replaced by gnrc_netapi_dispatch_receive() then threads will receive the GNRC_NETAPI_MSG_TYPE_RCV command instead, again with access to the Packet. NoteIf the data to be sent requires extra headers to be added for successful transmission (in the example, this would be IP and UDP headers), these have to be built manually before calling gnrc_netapi_dispatch_send(). In the interest of conciseness, this is omitted in this tutorial; please refer to gnrc_udp_hdr_build(), gnrc_ipv6_hdr_build() etc. for more information. GNRC is implemented according to the respective standards. So please note, that sending to a IPv6 link-local address always requires you by definition to also provide the interface you want to send to, otherwise address resolution might fail. How To Use Generic (GNRC) network stack is highly modular and can be adjusted to include only the desired features. In the following several of the available modules will be stated that you can include in your application's Makefile. To include the default network device(s) on your board: USEMODULE += netdev_default To auto-initialize these network devices as GNRC network interfaces USEMODULE += auto_init_gnrc_netif You may choose to build either as an IPv6 Node USEMODULE += gnrc_ipv6_default or as an IPv6 Router USEMODULE += gnrc_ipv6_router_default An IPv6 Router can forward packets, while an IPv6 Node will simply drop packets not targeted to it. If an IEEE 802.15.4 network device is present 6LoWPAN (with 6LoWPAN Fragmentation and IPv6 header compression (IPHC)) will be included automatically. For basic IPv6 (and 6LoWPAN) functionalities choose instead USEMODULE += gnrc_ipv6 or USEMODULE += gnrc_ipv6_router respectively. Those modules provide the bare minimum of IPv6 functionalities (no ICMPv6). Because of that, the Neighbor Information Base for IPv6 needs to be configured manually. If an IEEE 802.15.4 device is present 6LoWPAN will be included automatically, but no fragmentation or header compression support will be provided. For ICMPv6 echo request/reply (ping) functionality: USEMODULE += gnrc_icmpv6_echo For UDP support include USEMODULE += gnrc_udp To use UDP sock API with GNRC include USEMODULE += gnrc_sock_udp To include the RPL module USEMODULE += gnrc_rpl This will include the RPL module. To provide forwarding capabilities it is necessary to build the application with gnrc_ipv6_router_default (or gnrc_ipv6_router), not gnrc_ipv6_default (or gnrc_ipv6). Modules 6LoWPAN GNRC's 6LoWPAN implementation. Common MAC module A MAC module for providing common MAC parameters and helper functions. Dump Network Packets Dump network packets to STDOUT for debugging. Error reporting Allows for asynchronous error reporting in the network stack. GNRC LoRaWAN GNRC LoRaWAN stack implementation. GNRC communication interface Generic interface for IPC communication between GNRC modules. GNRC-specific implementation of the sock API Provides an implementation of the Sock API by the Generic (GNRC) network stack. GoMacH A traffic-adaptive multi-channel MAC. Helpers for
```


---
**Chunk 7** (Relevance: 0.306)
**Module:** gnrc
**Category:** core
**File:** group__net__gnrc__tcp.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\group__net__gnrc__tcp.html
**Section:** Unknown

```
() must have been successfully called. tcb must not be NULL. Parameters [in,out]tcbTCB holding the connection information. �� gnrc_tcp_ep_from_str() int gnrc_tcp_ep_from_str ( gnrc_tcp_ep_t * ep, const char * str ) Construct TCP connection endpoint from string. NoteThis function expects str in the IPv6 "URL" notation. The following strings specify a valid endpoint: [fe80::0a00:27ff:fe9f:7a5b%5]:8080 (with Port and Interface) [2001::0200:f8ff:fe21:67cf]:8080 (with Port) [2001::0200:f8ff:fe21:67cf] (addr only) Parameters [in,out]epEndpoint to initialize. [in]strString containing IPv6-Address to parse. Returns0 on success. -EINVAL if parsing of str failed. �� gnrc_tcp_ep_init() int gnrc_tcp_ep_init ( gnrc_tcp_ep_t * ep, int family, const uint8_t * addr, size_t addr_size, uint16_t port, uint16_t netif ) Initialize TCP connection endpoint. Parameters [in,out]epEndpoint to initialize. [in]familyAddress family of addr. [in]addrAddress for endpoint. [in]addr_sizeSize of addr. [in]portPort number for endpoint. [in]netifNetwork interface to use. Returns0 on success. -EAFNOSUPPORT if address_family is not supported. -EINVAL if addr_size does not match family. �� gnrc_tcp_get_local() int gnrc_tcp_get_local ( gnrc_tcp_tcb_t * tcb, gnrc_tcp_ep_t * ep ) Get the local end point of a connected TCB. Preconditiontcb must not be NULL ep must not be NULL Parameters [in]tcbTCB holding the connection information. [out]epThe local end point. Returns0 on success. -EADDRNOTAVAIL, when tcb in not in a connected state. �� gnrc_tcp_get_remote() int gnrc_tcp_get_remote ( gnrc_tcp_tcb_t * tcb, gnrc_tcp_ep_t * ep ) Get the remote end point of a connected TCB. Preconditiontcb must not be NULL ep must not be NULL Parameters [in]tcbTCB holding the connection information. [out]epThe remote end point. Returns0 on success. -ENOTCONN, when tcb in not in a connected state. �� gnrc_tcp_hdr_build() gnrc_pktsnip_t* gnrc_tcp_hdr_build ( gnrc_pktsnip_t * payload, uint16_t src, uint16_t dst ) Adds a TCP header to a given payload. Parameters [in]payloadPayload that follows the TCP header. [in]srcSource port number. [in]dstDestination port number. ReturnsNot NULL on success. NULL if TCP header was not allocated. �� gnrc_tcp_init() int gnrc_tcp_init ( void ) Initialize TCP. ReturnsPID of TCP thread on success -1 if TCB is already running. -EINVAL, if priority is greater than or equal SCHED_PRIO_LEVELS -EOVERFLOW, if there are too many threads running. �� gnrc_tcp_listen() int gnrc_tcp_listen ( gnrc_tcp_tcb_queue_t * queue, gnrc_tcp_tcb_t * tcbs, size_t tcbs_len, const gnrc_tcp_ep_t * local ) Configures a sequence of TCBs to wait for incoming connections. PreconditionAll TCBs behind tcbs must have been initialized via gnrc_tcp_tcb_init(). queue must not be NULL. tcbs must not be NULL. tcbs_len must be greater 0. local len must be NULL. local->port must not be
```


---
**Chunk 8** (Relevance: 0.302)
**Module:** gnrc
**Category:** core
**File:** gnrc__sock__internal_8h_source.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\gnrc__sock__internal_8h_source.html
**Section:** Unknown

```
_ctx)Create a sock internally. AF_INET6@ AF_INET6internetwork address family with IPv6: UDP, TCP, etc.Definition: af.h:39 gnrc_netif_itergnrc_netif_t * gnrc_netif_iter(const gnrc_netif_t *prev)Iterate over all network interfaces. gnrc_netif_highlanderstatic bool gnrc_netif_highlander(void)Check if there can only be one gnrc_netif_t interface.Definition: netif.h:440 gnrc_nettype_tgnrc_nettype_tDefinition of protocol types in the network stack.Definition: nettype.h:51 SOCK_ADDR_ANY_NETIF#define SOCK_ADDR_ANY_NETIFSpecial netif ID for "any interface".Definition: sock.h:153 ip.hRaw IPv4/IPv6 sock definitions. mbox.hMailbox API. netreg.hDefinitions to register network protocol PIDs to use with GNRC communication interface. portrange.hService Name and Transport Protocol Port Number Registry. gnrc_netif_tRepresentation of a network interface.Definition: netif.h:135 gnrc_netif_t::pidkernel_pid_t pidPID of the network interface's thread.Definition: netif.h:226 gnrc_pktsnipType to represent parts (either headers or payload) of a packet, called snips.Definition: pkt.h:108 gnrc_sock_recv_aux_tStructure to retrieve auxiliary data from gnrc_sock_recv.Definition: gnrc_sock_internal.h:72 gnrc_sock_recv_aux_t::flagsuint8_t flagsFlags.Definition: gnrc_sock_internal.h:90 gnrc_sock_recv_aux_t::localsock_ip_ep_t * locallocal IP address PDU was received onDefinition: gnrc_sock_internal.h:79 gnrc_sock_recv_aux_t::timestampuint64_t * timestamptimestamp PDU was received at in nanosecondsDefinition: gnrc_sock_internal.h:82 gnrc_sock_recv_aux_t::rssiint16_t * rssiRSSI value of received PDU.Definition: gnrc_sock_internal.h:85 gnrc_sock_regsock Network protocol registry infoDefinition: sock_types.h:90 sock_ip_ep_tAbstract IP end point and end point for a raw IP sock object.Definition: sock.h:177 sock_ip_ep_t::netifuint16_t netifstack-specific network interface IDDefinition: sock.h:208 sock_ip_ep_t::addrunion sock_ip_ep_t::@388 addraddress sock_types.hGNRC-specific types and function definitions. Generated on Sat Jul 12 2025 00:23:29 by 1.9.1
```


---

**Instructions:**
1. Use the provided RIOT OS documentation chunks to answer the user's question comprehensively
2. Focus on practical, actionable information from the documentation
3. Reference specific modules, functions, or APIs mentioned in the chunks when relevant
4. If the chunks don't contain sufficient information, clearly state what information is missing
5. Provide code examples or configuration snippets when available in the chunks
6. Maintain technical accuracy and use RIOT OS terminology correctly
7. Structure your response clearly with headers and bullet points when appropriate

**Response Guidelines:**
- Start with a direct answer to the user's question
- Provide step-by-step instructions when applicable
- Include relevant code examples from the documentation
- Mention any prerequisites or dependencies
- Reference specific file paths or modules when helpful
- End with related concepts or next steps if appropriate

Please provide a comprehensive answer based on the RIOT OS documentation provided above.

I then ran it in request.py and got this:

Top 8 results for: I want to make a udp server that returns a resource using gnrc

[1] Score: 0.9133 (cosine: 0.7133, path_boost: 0.2000)
🎯 Path keywords matched: udp, gnrc
📁 File: RIOT/examples/networking/gnrc/gnrc_networking_mac/udp.c
🔖 Chunk: udp_chunk0003
📄 Full Text:
----------------------------------------
uint16_t port;

    /* check if server is already running */
    if (server.target.pid != KERNEL_PID_UNDEF) {
        printf("Error: server already running on port %" PRIu32 "\n",
               server.demux_ctx);
        return;
    }
    /* parse port */
    port = atoi(port_str);
    if (port == 0) {
        puts("Error: invalid port specified");
        return;
    }
    /* start server (which means registering pktdump for the chosen port) */
    server.target.pid = gnrc_pktdump_pid;
    server.demux_ctx = (uint32_t)port;
    gnrc_netreg_register(GNRC_NETTYPE_UDP, &server);
    printf("Success: started UDP server on port %" PRIu16 "\n", port);
}
----------------------------------------
--------------------------------------------------------------------------------

[2] Score: 0.8763 (cosine: 0.6763, path_boost: 0.2000)
🎯 Path keywords matched: udp, gnrc
📁 File: RIOT/examples/networking/gnrc/gnrc_networking_mac/udp.c
🔖 Chunk: udp_chunk0001
📄 Full Text:
----------------------------------------
/*
 * Copyright (C) 2015-17 Freie Universität Berlin
 *
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v2.1. See the file LICENSE in the top level
 * directory for more details.
 */

/**
 * @ingroup     examples
 * @{
 *
 * @file
 * @brief       Demonstrating the sending and receiving of UDP data
 *
 * @author      Hauke Petersen <hauke.petersen@fu-berlin.de>
 * @author      Martine Lenders <m.lenders@fu-berlin.de>
 *
 * @}
 */

#include <stdio.h>
#include <inttypes.h>

#include "net/gnrc.h"
#include "net/gnrc/ipv6.h"
#include "net/gnrc/netif.h"
#include "net/gnrc/netif/hdr.h"
#include "net/gnrc/udp.h"
#include "net/gnrc/pktdump.h"
#include "shell.h"
#include "timex.h"
#include "utlist.h"
#include "xtimer.h"

static gnrc_netreg_entry_t server = GNRC_NETREG_ENTRY_INIT_PID(GNRC_NETREG_DEMUX_CTX_ALL,
                                                               KERNEL_PID_UNDEF);

static void send(char *addr_str, char *port_str, char *data, unsigned int num,
                 unsigned int delay)
{
    gnrc_netif_t *netif = NULL;
    char *iface;
    uint16_t port;
    ipv6_addr_t addr;

    /* get interface, if available */
    iface = ipv6_addr_split_iface(addr_str);
    if ((!iface) && (gnrc_netif_numof() == 1)) {
        netif = gnrc_netif_iter(NULL);
    }
    else if (iface){
        netif = gnrc_netif_get_by_pid(atoi(iface));
    }
    /* parse destination address */
    if (ipv6_addr_from_str(&addr, addr_str) == NULL) {
        puts("Error: unable to parse destination address");
        return;
    }
    /* parse port */
    port = atoi(port_str);
    if (port == 0) {
        puts("Error: unable to parse destination port");
        return;
    }
----------------------------------------
--------------------------------------------------------------------------------

[3] Score: 0.8420 (cosine: 0.6420, path_boost: 0.2000)
🎯 Path keywords matched: udp, gnrc
📁 File: RIOT/examples/networking/gnrc/gnrc_networking_mac/udp.c
🔖 Chunk: udp_chunk0002
📄 Full Text:
----------------------------------------
for (unsigned int i = 0; i < num; i++) {
        gnrc_pktsnip_t *payload, *udp, *ip;
        unsigned payload_size;
        /* allocate payload */
        payload = gnrc_pktbuf_add(NULL, data, strlen(data), GNRC_NETTYPE_UNDEF);
        if (payload == NULL) {
            puts("Error: unable to copy data to packet buffer");
            return;
        }
        /* store size for output */
        payload_size = (unsigned)payload->size;
        /* allocate UDP header, set source port := destination port */
        udp = gnrc_udp_hdr_build(payload, port, port);
        if (udp == NULL) {
            puts("Error: unable to allocate UDP header");
            gnrc_pktbuf_release(payload);
            return;
        }
        /* allocate IPv6 header */
        ip = gnrc_ipv6_hdr_build(udp, NULL, &addr);
        if (ip == NULL) {
            puts("Error: unable to allocate IPv6 header");
            gnrc_pktbuf_release(udp);
            return;
        }
        /* add netif header, if interface was given */
        if (netif != NULL) {
            gnrc_pktsnip_t *netif_hdr = gnrc_netif_hdr_build(NULL, 0, NULL, 0);
            if (netif_hdr == NULL) {
                puts("Error: unable to allocate netif header");
                gnrc_pktbuf_release(ip);
                return;
            }
            gnrc_netif_hdr_set_netif(netif_hdr->data, netif);
            ip = gnrc_pkt_prepend(ip, netif_hdr);
        }
        /* send packet */
        if (!gnrc_netapi_dispatch_send(GNRC_NETTYPE_UDP, GNRC_NETREG_DEMUX_CTX_ALL, ip)) {
            puts("Error: unable to locate UDP thread");
            gnrc_pktbuf_release(ip);
            return;
        }
        /* access to `payload` was implicitly given up with the send operation above
         * => use temporary variable for output */
        printf("Success: sent %u byte(s) to [%s]:%u\n", payload_size, addr_str,
               port);
        xtimer_usleep(delay);
    }
}
----------------------------------------
--------------------------------------------------------------------------------

[4] Score: 0.8356 (cosine: 0.6356, path_boost: 0.2000)
🎯 Path keywords matched: udp, gnrc
📁 File: RIOT/examples/networking/gnrc/gnrc_networking_mac/udp.c
🔖 Chunk: udp_chunk0004
📄 Full Text:
----------------------------------------
/* check if server is running at all */
    if (server.target.pid == KERNEL_PID_UNDEF) {
        printf("Error: server was not running\n");
        return;
    }
    /* stop server */
    gnrc_netreg_unregister(GNRC_NETTYPE_UDP, &server);
    server.target.pid = KERNEL_PID_UNDEF;
    puts("Success: stopped UDP server");
}
----------------------------------------
--------------------------------------------------------------------------------

[5] Score: 0.7088 (cosine: 0.6088, path_boost: 0.1000)
🎯 Path keywords matched: gnrc
📁 File: RIOT/examples/networking/gnrc/gnrc_networking/README.md
🔖 Chunk: README_chunk0001
📄 Full Text:
----------------------------------------
# gnrc_networking example

This example shows you how to try out the code in two different ways:
Either by communicating between the RIOT machine and its Linux host,
or by communicating between two RIOT instances.
Note that the former only works with native, i.e. if you run RIOT on
your Linux machine.

## Connecting RIOT native and the Linux host

> **Note:** RIOT does not support IPv4, so you need to stick to IPv6
> anytime. To establish a connection between RIOT and the Linux host,
> you will need `netcat` (with IPv6 support). Ubuntu 14.04 comes with
> netcat IPv6 support pre-installed.
> On Debian it's available in the package `netcat-openbsd`. Be aware
> that many programs require you to add an option such as -6 to tell
> them to use IPv6, otherwise they will fail. If you're using a
> _Raspberry Pi_, run `sudo modprobe ipv6` before trying this example,
> because raspbian does not load the IPv6 module automatically.
> On some systems (openSUSE for example), the _firewall_ may interfere,
> and prevent some packets to arrive at the application (they will
> however show up in Wireshark, which can be confusing). So be sure to
> adjust your firewall rules, or turn it off (who needs security
> anyway).

First, make sure you've compiled the application by calling `make`.

Now, create a tap interface:

    sudo ip tuntap add tap0 mode tap user ${USER}
    sudo ip link set tap0 up

Now you can start the `gnrc_networking` example by invoking `make term`.
This should automatically connect to the `tap0` interface. If this
doesn't work for any reason, run make term with the tap0 interface as
the PORT environment variable:

    PORT=tap0 make term

To verify that there is connectivity between RIOT and Linux, go to the
RIOT console and run `ifconfig`:
----------------------------------------
--------------------------------------------------------------------------------

[6] Score: 0.6766 (cosine: 0.5766, path_boost: 0.1000)
🎯 Path keywords matched: udp
📁 File: RIOT/examples/networking/misc/posix_sockets/udp.c
🔖 Chunk: udp_chunk0001
📄 Full Text:
----------------------------------------
/*
 * Copyright (C) 2015 Martine Lenders <mlenders@inf.fu-berlin.de>
 *
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v2.1. See the file LICENSE in the top level
 * directory for more details.
 */

/**
 * @ingroup     examples
 * @{
 *
 * @file
 * @brief       Demonstrating the sending and receiving of UDP data over POSIX sockets.
 *
 * @author      Martine Lenders <mlenders@inf.fu-berlin.de>
 *
 * @}
 */

/* needed for posix usleep */
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#ifdef SOCK_HAS_IPV6
#include "net/ipv6/addr.h"     /* for interface parsing */
#include "net/netif.h"         /* for resolving ipv6 scope */
#endif /* SOCK_HAS_IPV6 */

#include "shell.h"
#include "thread.h"

#define SERVER_MSG_QUEUE_SIZE   (8)
#define SERVER_BUFFER_SIZE      (64)

static int server_socket = -1;
static char server_buffer[SERVER_BUFFER_SIZE];
static char server_stack[THREAD_STACKSIZE_DEFAULT];
static msg_t server_msg_queue[SERVER_MSG_QUEUE_SIZE];
----------------------------------------
--------------------------------------------------------------------------------

[7] Score: 0.6527 (cosine: 0.5527, path_boost: 0.1000)
🎯 Path keywords matched: udp
📁 File: RIOT/examples/networking/misc/posix_sockets/udp.c
🔖 Chunk: udp_chunk0004
📄 Full Text:
----------------------------------------
/* check if server is already running */
    if (server_socket >= 0) {
        puts("Error: server already running");
        return 1;
    }
    /* start server (which means registering pktdump for the chosen port) */
    if (thread_create(server_stack, sizeof(server_stack), THREAD_PRIORITY_MAIN - 1,
                      0,
                      _server_thread, port_str, "UDP server") <= KERNEL_PID_UNDEF) {
        server_socket = -1;
        puts("error initializing thread");
        return 1;
    }
    return 0;
}
----------------------------------------
--------------------------------------------------------------------------------

[8] Score: 0.6350 (cosine: 0.5350, path_boost: 0.1000)
🎯 Path keywords matched: udp
📁 File: RIOT/examples/networking/misc/posix_sockets/udp.c
🔖 Chunk: udp_chunk0005
📄 Full Text:
----------------------------------------
if (argc < 2) {
        printf("usage: %s [send|server]\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "send") == 0) {
        uint32_t num = 1;
        uint32_t delay = 1000000;
        if (argc < 5) {
            printf("usage: %s send <addr> <port> <data> [<num> [<delay in us>]]\n",
                   argv[0]);
            return 1;
        }
        if (argc > 5) {
            num = atoi(argv[5]);
        }
        if (argc > 6) {
            delay = atoi(argv[6]);
        }
        return udp_send(argv[2], argv[3], argv[4], num, delay);
    }
    else if (strcmp(argv[1], "server") == 0) {
        if (argc < 3) {
            printf("usage: %s server [start|stop]\n", argv[0]);
            return 1;
        }
        if (strcmp(argv[2], "start") == 0) {
            if (argc < 4) {
                printf("usage %s server start <port>\n", argv[0]);
                return 1;
            }
            return udp_start_server(argv[3]);
        }
        else {
            puts("error: invalid command");
            return 1;
        }
    }
    else {
        puts("error: invalid command");
        return 1;
    }
}

SHELL_COMMAND(udp, "send data over UDP and listen on UDP ports", _udp_cmd);

/** @} */


I arbitrarily combined these two result to form the following prompt I later gave to ClaudeYou are a RIOT OS expert assistant. Use the following documentation chunks to answer the user's question about RIOT OS.

**User Question:** I want to make a udp server that returns a resource using gnrc

**RIOT OS Documentation Context:**
The following 8 chunks were retrieved from the official RIOT OS documentation based on semantic similarity and relevance to your question:


---
**Chunk 1** (Relevance: 0.399)
**Module:** gnrc
**Category:** core
**File:** gnrc_2udp_8h.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\gnrc_2udp_8h.html
**Section:** Unknown

```
sys/include/net/gnrc/udp.h File Reference Toggle navigation Documentation The friendly Operating System for the Internet of Things Macros | Functions udp.h File ReferenceNetworking » Generic (GNRC) network stack » UDP UDP GNRC definition. More... Detailed Description UDP GNRC definition. AuthorHauke Petersen hauke.nosp@m..pet.nosp@m.ersen.nosp@m.@fu-.nosp@m.berli.nosp@m.n.de Definition in file udp.h. #include <stdint.h> #include "byteorder.h" #include "net/gnrc.h" #include "net/udp.h" Include dependency graph for udp.h: This browser is not able to show SVG: try Firefox, Chrome, Safari, or Opera instead. This graph shows which files directly or indirectly include this file: This browser is not able to show SVG: try Firefox, Chrome, Safari, or Opera instead. Go to the source code of this file. Macros #define CONFIG_GNRC_UDP_MSG_QUEUE_SIZE_EXP   (3U) Default message queue size for the UDP thread (as exponent of 2^n). More... #define GNRC_UDP_PRIO   (THREAD_PRIORITY_MAIN - 2) Priority of the UDP thread. #define GNRC_UDP_STACK_SIZE   ((THREAD_STACKSIZE_SMALL) - 64) Default stack size to use for the UDP thread. More... #define GNRC_UDP_MSG_QUEUE_SIZE   (1 << CONFIG_GNRC_UDP_MSG_QUEUE_SIZE_EXP) Message queue size to use for the UDP thread. Functions int gnrc_udp_calc_csum (gnrc_pktsnip_t *hdr, gnrc_pktsnip_t *pseudo_hdr) Calculate the checksum for the given packet. More... gnrc_pktsnip_t * gnrc_udp_hdr_build (gnrc_pktsnip_t *payload, uint16_t src, uint16_t dst) Allocate and initialize a fresh UDP header in the packet buffer. More... int gnrc_udp_init (void) Initialize and start UDP. More... Generated on Sat Jul 12 2025 00:26:07 by 1.9.1
```


---
**Chunk 2** (Relevance: 0.348)
**Module:** udp
**Category:** sys
**File:** group__net__sock__udp.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\group__net__sock__udp.html
**Section:** Unknown

```
UDP sock API Toggle navigation Documentation The friendly Operating System for the Internet of Things Files | Data Structures | Typedefs | Functions UDP sock APINetworking » Sock API Sock submodule for UDP. More... Detailed Description Sock submodule for UDP. How To Use First you need to include a module that implements this API in your application's Makefile. For example the implementation for GNRC is called gnrc_sock_udp. A Simple UDP Echo Server #include <stdio.h> #include "net/sock/udp.h" uint8_t buf[128]; int main(void) { sock_udp_ep_t local = SOCK_IPV6_EP_ANY; sock_udp_t sock; local.port = 12345; if (sock_udp_create(&sock, &local, NULL, 0) < 0) { puts("Error creating UDP sock"); return 1; } while (1) { sock_udp_ep_t remote; ssize_t res; if ((res = sock_udp_recv(&sock, buf, sizeof(buf), SOCK_NO_TIMEOUT, &remote)) >= 0) { puts("Received a message"); if (sock_udp_send(&sock, buf, res, &remote) < 0) { puts("Error sending reply"); } } } return 0; } sock_udp_createint sock_udp_create(sock_udp_t *sock, const sock_udp_ep_t *local, const sock_udp_ep_t *remote, uint16_t flags)Creates a new UDP sock object. sock_udp_sendstatic ssize_t sock_udp_send(sock_udp_t *sock, const void *data, size_t len, const sock_udp_ep_t *remote)Sends a UDP message to remote end point.Definition: udp.h:754 sock_udp_recvstatic ssize_t sock_udp_recv(sock_udp_t *sock, void *data, size_t max_len, uint32_t timeout, sock_udp_ep_t *remote)Receives a UDP message from a remote end point.Definition: udp.h:530 SOCK_IPV6_EP_ANY#define SOCK_IPV6_EP_ANYAddress to bind to any IPv6 address.Definition: sock.h:165 SOCK_NO_TIMEOUT#define SOCK_NO_TIMEOUTSpecial value meaning "wait forever" (don't timeout)Definition: sock.h:172 udp.hUDP sock definitions. _sock_tl_epCommon IP-based transport layer end point.Definition: sock.h:214 _sock_tl_ep::portuint16_t porttransport layer port (in host byte order)Definition: sock.h:246 sock_udpUDP sock type.Definition: sock_types.h:128 Above you see a simple UDP echo server. Don't forget to also include the IPv6 module of your networking implementation (e.g. gnrc_ipv6_default for Generic (GNRC) network stack GNRC) and at least one network device. After including the header file for UDP sock, we create some buffer space buf to store the data received by the server: #include "net/sock/udp.h" uint8_t buf[128]; To be able to listen for incoming packets we bind the sock by setting a local end point with a port (12345 in this case). We then proceed to create the sock. It is bound to local and thus listens for UDP packets with destination port 12345. Since we don't need any further configuration we set the flags to 0. In case of an error we stop the program: sock_udp_ep_t local = SOCK_IPV6_EP_ANY; sock_udp_t sock; local.port = 12345; if (sock_udp_create(&sock, &local, NULL, 0) < 0) { puts("Error creating UDP sock"); return 1; } The application then waits indefinitely for an incoming message in buf from remote. If we want to timeout this wait period we could alternatively set the timeout parameter of sock_udp_recv() to a value != SOCK_NO_TIMEOUT. If an error occurs on receive we just ignore it and continue looping. If we
```


---
**Chunk 3** (Relevance: 0.332)
**Module:** gnrc
**Category:** sys
**File:** group__net__gnrc__nettype__udp.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\group__net__gnrc__nettype__udp.html
**Section:** Unknown

```
gnrc_nettype_udp Toggle navigation Documentation The friendly Operating System for the Internet of Things gnrc_nettype_udpNetworking » Generic (GNRC) network stack | Generic pseudomodules » gnrc_nettype: Protocol type Enables GNRC_NETTYPE_UDP. Enables GNRC_NETTYPE_UDP. Generated on Sat Jul 12 2025 00:28:03 by 1.9.1
```


---
**Chunk 4** (Relevance: 0.316)
**Module:** gnrc
**Category:** sys
**File:** group__net__sock__async__event.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\group__net__sock__async__event.html
**Section:** Unknown

```
_udp_recv(sock_udp_t *sock, void *data, size_t max_len, uint32_t timeout, sock_udp_ep_t *remote)Receives a UDP message from a remote end point.Definition: udp.h:530 SOCK_IPV6_EP_ANY#define SOCK_IPV6_EP_ANYAddress to bind to any IPv6 address.Definition: sock.h:165 event_loopstatic void event_loop(event_queue_t *queue)Simple event loop.Definition: event.h:486 event_queue_initstatic void event_queue_init(event_queue_t *queue)Initialize an event queue.Definition: event.h:184 event.hAsynchronous sock using Event Queue definitions. udp.hUDP sock definitions. PTRTAGevent queue structureDefinition: event.h:153 _sock_tl_epCommon IP-based transport layer end point.Definition: sock.h:214 _sock_tl_ep::portuint16_t porttransport layer port (in host byte order)Definition: sock.h:246 sock_udpUDP sock type.Definition: sock_types.h:128 Above you see a simple UDP echo server using Event Queue. Don't forget to also include the IPv6 module of your networking implementation (e.g. gnrc_ipv6_default for Generic (GNRC) network stack GNRC) and at least one network device. After including the header file for UDP sock and asynchronous handling, we create the event queue queue and allocate some buffer space buf to store the data received by the server: #include "net/sock/udp.h" #include "net/sock/async/event.h" event_queue_t queue; uint8_t buf[128]; We then define an event handler in form of a function. The event handler checks if the triggering event was a receive event by checking the flags provided with sock_event_t::type. If it is a receive event it copies the incoming message to buf and its sender into remote using sock_udp_recv(). Note, that we use sock_udp_recv() non-blocking by setting timeout to 0. If an error occurs on receive, we just ignore it and return from the event handler. If we receive a message we use its remote to reply. In case of an error on send, we print an according message: void handler(sock_udp_t *sock, sock_async_flags_t type, void *arg) { (void)arg; if (type & SOCK_ASYNC_MSG_RECV) { sock_udp_ep_t remote; ssize_t res; if ((res = sock_udp_recv(sock, buf, sizeof(buf), 0, &remote)) >= 0) { puts("Received a message"); if (sock_udp_send(sock, buf, res, &remote) < 0) { puts("Error sending reply"); } } } } To be able to listen for incoming packets we bind the sock by setting a local end point with a port (12345 in this case). We then proceed to create the sock. It is bound to local and thus listens for UDP packets with destination port 12345. Since we don't need any further configuration we set the flags to 0. In case of an error we stop the program: sock_udp_ep_t local = SOCK_IPV6_EP_ANY; sock_udp_t sock; local.port = 12345; if (sock_udp_create(&sock, &local, NULL, 0) < 0) { puts("Error creating UDP sock"); return 1; } Finally, we initialize the event queue queue, initialize asynchronous event handling for sock using it and the previously defined event handler, and go into an endless loop to handle all occurring events on queue: event_queue_init(&queue); sock_udp_event_init(&sock, &queue, handler, NULL); event_loop(&queue); Files file  event.h Asynchronous sock using Event Queue definitions. file  sock_async_ctx.h Type definitions for asynchronous socks with Event Queue. Data Structures union  sock_event_cb_t Generalized callback type. More... struct  sock_event_t Event definition
```


---
**Chunk 5** (Relevance: 0.314)
**Module:** gnrc
**Category:** core
**File:** group__net__gnrc__udp.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\group__net__gnrc__udp.html
**Section:** Unknown

```
UDP Toggle navigation Documentation The friendly Operating System for the Internet of Things Modules | Files | Macros | Functions UDPNetworking » Generic (GNRC) network stack GNRC's implementation of the UDP protocol. More... Detailed Description GNRC's implementation of the UDP protocol. Modules GNRC UDP compile configurations Files file  udp.h UDP GNRC definition. Macros #define GNRC_UDP_MSG_QUEUE_SIZE   (1 << CONFIG_GNRC_UDP_MSG_QUEUE_SIZE_EXP) Message queue size to use for the UDP thread. Functions int gnrc_udp_calc_csum (gnrc_pktsnip_t *hdr, gnrc_pktsnip_t *pseudo_hdr) Calculate the checksum for the given packet. More... gnrc_pktsnip_t * gnrc_udp_hdr_build (gnrc_pktsnip_t *payload, uint16_t src, uint16_t dst) Allocate and initialize a fresh UDP header in the packet buffer. More... int gnrc_udp_init (void) Initialize and start UDP. More... Function Documentation �� gnrc_udp_calc_csum() int gnrc_udp_calc_csum ( gnrc_pktsnip_t * hdr, gnrc_pktsnip_t * pseudo_hdr ) Calculate the checksum for the given packet. Parameters [in]hdrPointer to the UDP header [in]pseudo_hdrPointer to the network layer header Returns0 on success -EBADMSG if hdr is not of type GNRC_NETTYPE_UDP -EFAULT if hdr or pseudo_hdr is NULL -ENOENT if gnrc_pktsnip_t::type of pseudo_hdr is not known �� gnrc_udp_hdr_build() gnrc_pktsnip_t* gnrc_udp_hdr_build ( gnrc_pktsnip_t * payload, uint16_t src, uint16_t dst ) Allocate and initialize a fresh UDP header in the packet buffer. Parameters [in]payloadPayload contained in the UDP packet [in]srcSource port in host byte order [in]dstDestination port in host byte order Returnspointer to the newly created (and allocated) header NULL on src == NULL, dst == NULL, src_len != 2, dst_len != 2 or on allocation error Preconditionsrc > 0 and dst > 0 �� gnrc_udp_init() int gnrc_udp_init ( void ) Initialize and start UDP. ReturnsPID of the UDP thread negative value on error Generated on Sat Jul 12 2025 00:27:55 by 1.9.1
```


---
**Chunk 6** (Relevance: 0.307)
**Module:** gnrc
**Category:** core
**File:** group__net__gnrc.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\group__net__gnrc.html
**Section:** Unknown

```
 to be sent is added to the packet buffer. This ensures its intactness during the sending process. After the data to be sent has been added to the packet buffer, its parent data structure can safely be freed or reused. Then, the pkt will be sent to all threads that registered for GNRC_NETTYPE_UDP and the demux context 80. Every registered thread will receive a GNRC_NETAPI_MSG_TYPE_SND command and can access the Packet. Note that at this point, the threads receiving pkt act as its owners, so please don't modify pkt after calling any dispatch function. If gnrc_netapi_dispatch_send() is replaced by gnrc_netapi_dispatch_receive() then threads will receive the GNRC_NETAPI_MSG_TYPE_RCV command instead, again with access to the Packet. NoteIf the data to be sent requires extra headers to be added for successful transmission (in the example, this would be IP and UDP headers), these have to be built manually before calling gnrc_netapi_dispatch_send(). In the interest of conciseness, this is omitted in this tutorial; please refer to gnrc_udp_hdr_build(), gnrc_ipv6_hdr_build() etc. for more information. GNRC is implemented according to the respective standards. So please note, that sending to a IPv6 link-local address always requires you by definition to also provide the interface you want to send to, otherwise address resolution might fail. How To Use Generic (GNRC) network stack is highly modular and can be adjusted to include only the desired features. In the following several of the available modules will be stated that you can include in your application's Makefile. To include the default network device(s) on your board: USEMODULE += netdev_default To auto-initialize these network devices as GNRC network interfaces USEMODULE += auto_init_gnrc_netif You may choose to build either as an IPv6 Node USEMODULE += gnrc_ipv6_default or as an IPv6 Router USEMODULE += gnrc_ipv6_router_default An IPv6 Router can forward packets, while an IPv6 Node will simply drop packets not targeted to it. If an IEEE 802.15.4 network device is present 6LoWPAN (with 6LoWPAN Fragmentation and IPv6 header compression (IPHC)) will be included automatically. For basic IPv6 (and 6LoWPAN) functionalities choose instead USEMODULE += gnrc_ipv6 or USEMODULE += gnrc_ipv6_router respectively. Those modules provide the bare minimum of IPv6 functionalities (no ICMPv6). Because of that, the Neighbor Information Base for IPv6 needs to be configured manually. If an IEEE 802.15.4 device is present 6LoWPAN will be included automatically, but no fragmentation or header compression support will be provided. For ICMPv6 echo request/reply (ping) functionality: USEMODULE += gnrc_icmpv6_echo For UDP support include USEMODULE += gnrc_udp To use UDP sock API with GNRC include USEMODULE += gnrc_sock_udp To include the RPL module USEMODULE += gnrc_rpl This will include the RPL module. To provide forwarding capabilities it is necessary to build the application with gnrc_ipv6_router_default (or gnrc_ipv6_router), not gnrc_ipv6_default (or gnrc_ipv6). Modules 6LoWPAN GNRC's 6LoWPAN implementation. Common MAC module A MAC module for providing common MAC parameters and helper functions. Dump Network Packets Dump network packets to STDOUT for debugging. Error reporting Allows for asynchronous error reporting in the network stack. GNRC LoRaWAN GNRC LoRaWAN stack implementation. GNRC communication interface Generic interface for IPC communication between GNRC modules. GNRC-specific implementation of the sock API Provides an implementation of the Sock API by the Generic (GNRC) network stack. GoMacH A traffic-adaptive multi-channel MAC. Helpers for
```


---
**Chunk 7** (Relevance: 0.306)
**Module:** gnrc
**Category:** core
**File:** group__net__gnrc__tcp.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\group__net__gnrc__tcp.html
**Section:** Unknown

```
() must have been successfully called. tcb must not be NULL. Parameters [in,out]tcbTCB holding the connection information. �� gnrc_tcp_ep_from_str() int gnrc_tcp_ep_from_str ( gnrc_tcp_ep_t * ep, const char * str ) Construct TCP connection endpoint from string. NoteThis function expects str in the IPv6 "URL" notation. The following strings specify a valid endpoint: [fe80::0a00:27ff:fe9f:7a5b%5]:8080 (with Port and Interface) [2001::0200:f8ff:fe21:67cf]:8080 (with Port) [2001::0200:f8ff:fe21:67cf] (addr only) Parameters [in,out]epEndpoint to initialize. [in]strString containing IPv6-Address to parse. Returns0 on success. -EINVAL if parsing of str failed. �� gnrc_tcp_ep_init() int gnrc_tcp_ep_init ( gnrc_tcp_ep_t * ep, int family, const uint8_t * addr, size_t addr_size, uint16_t port, uint16_t netif ) Initialize TCP connection endpoint. Parameters [in,out]epEndpoint to initialize. [in]familyAddress family of addr. [in]addrAddress for endpoint. [in]addr_sizeSize of addr. [in]portPort number for endpoint. [in]netifNetwork interface to use. Returns0 on success. -EAFNOSUPPORT if address_family is not supported. -EINVAL if addr_size does not match family. �� gnrc_tcp_get_local() int gnrc_tcp_get_local ( gnrc_tcp_tcb_t * tcb, gnrc_tcp_ep_t * ep ) Get the local end point of a connected TCB. Preconditiontcb must not be NULL ep must not be NULL Parameters [in]tcbTCB holding the connection information. [out]epThe local end point. Returns0 on success. -EADDRNOTAVAIL, when tcb in not in a connected state. �� gnrc_tcp_get_remote() int gnrc_tcp_get_remote ( gnrc_tcp_tcb_t * tcb, gnrc_tcp_ep_t * ep ) Get the remote end point of a connected TCB. Preconditiontcb must not be NULL ep must not be NULL Parameters [in]tcbTCB holding the connection information. [out]epThe remote end point. Returns0 on success. -ENOTCONN, when tcb in not in a connected state. �� gnrc_tcp_hdr_build() gnrc_pktsnip_t* gnrc_tcp_hdr_build ( gnrc_pktsnip_t * payload, uint16_t src, uint16_t dst ) Adds a TCP header to a given payload. Parameters [in]payloadPayload that follows the TCP header. [in]srcSource port number. [in]dstDestination port number. ReturnsNot NULL on success. NULL if TCP header was not allocated. �� gnrc_tcp_init() int gnrc_tcp_init ( void ) Initialize TCP. ReturnsPID of TCP thread on success -1 if TCB is already running. -EINVAL, if priority is greater than or equal SCHED_PRIO_LEVELS -EOVERFLOW, if there are too many threads running. �� gnrc_tcp_listen() int gnrc_tcp_listen ( gnrc_tcp_tcb_queue_t * queue, gnrc_tcp_tcb_t * tcbs, size_t tcbs_len, const gnrc_tcp_ep_t * local ) Configures a sequence of TCBs to wait for incoming connections. PreconditionAll TCBs behind tcbs must have been initialized via gnrc_tcp_tcb_init(). queue must not be NULL. tcbs must not be NULL. tcbs_len must be greater 0. local len must be NULL. local->port must not be
```


---
**Chunk 8** (Relevance: 0.302)
**Module:** gnrc
**Category:** core
**File:** gnrc__sock__internal_8h_source.html
**Path:** C:\Users\asums\OneDrive\Desktop\RIOT\doc\doxygen\html\gnrc__sock__internal_8h_source.html
**Section:** Unknown

```
_ctx)Create a sock internally. AF_INET6@ AF_INET6internetwork address family with IPv6: UDP, TCP, etc.Definition: af.h:39 gnrc_netif_itergnrc_netif_t * gnrc_netif_iter(const gnrc_netif_t *prev)Iterate over all network interfaces. gnrc_netif_highlanderstatic bool gnrc_netif_highlander(void)Check if there can only be one gnrc_netif_t interface.Definition: netif.h:440 gnrc_nettype_tgnrc_nettype_tDefinition of protocol types in the network stack.Definition: nettype.h:51 SOCK_ADDR_ANY_NETIF#define SOCK_ADDR_ANY_NETIFSpecial netif ID for "any interface".Definition: sock.h:153 ip.hRaw IPv4/IPv6 sock definitions. mbox.hMailbox API. netreg.hDefinitions to register network protocol PIDs to use with GNRC communication interface. portrange.hService Name and Transport Protocol Port Number Registry. gnrc_netif_tRepresentation of a network interface.Definition: netif.h:135 gnrc_netif_t::pidkernel_pid_t pidPID of the network interface's thread.Definition: netif.h:226 gnrc_pktsnipType to represent parts (either headers or payload) of a packet, called snips.Definition: pkt.h:108 gnrc_sock_recv_aux_tStructure to retrieve auxiliary data from gnrc_sock_recv.Definition: gnrc_sock_internal.h:72 gnrc_sock_recv_aux_t::flagsuint8_t flagsFlags.Definition: gnrc_sock_internal.h:90 gnrc_sock_recv_aux_t::localsock_ip_ep_t * locallocal IP address PDU was received onDefinition: gnrc_sock_internal.h:79 gnrc_sock_recv_aux_t::timestampuint64_t * timestamptimestamp PDU was received at in nanosecondsDefinition: gnrc_sock_internal.h:82 gnrc_sock_recv_aux_t::rssiint16_t * rssiRSSI value of received PDU.Definition: gnrc_sock_internal.h:85 gnrc_sock_regsock Network protocol registry infoDefinition: sock_types.h:90 sock_ip_ep_tAbstract IP end point and end point for a raw IP sock object.Definition: sock.h:177 sock_ip_ep_t::netifuint16_t netifstack-specific network interface IDDefinition: sock.h:208 sock_ip_ep_t::addrunion sock_ip_ep_t::@388 addraddress sock_types.hGNRC-specific types and function definitions. Generated on Sat Jul 12 2025 00:23:29 by 1.9.1
```


Here are also 8 Chunks directly from the repository’s examples:

[1] Score: 0.9133 (cosine: 0.7133, path_boost: 0.2000)
🎯 Path keywords matched: udp, gnrc
📁 File: RIOT/examples/networking/gnrc/gnrc_networking_mac/udp.c
🔖 Chunk: udp_chunk0003
📄 Full Text:
----------------------------------------
uint16_t port;

    /* check if server is already running */
    if (server.target.pid != KERNEL_PID_UNDEF) {
        printf("Error: server already running on port %" PRIu32 "\n",
               server.demux_ctx);
        return;
    }
    /* parse port */
    port = atoi(port_str);
    if (port == 0) {
        puts("Error: invalid port specified");
        return;
    }
    /* start server (which means registering pktdump for the chosen port) */
    server.target.pid = gnrc_pktdump_pid;
    server.demux_ctx = (uint32_t)port;
    gnrc_netreg_register(GNRC_NETTYPE_UDP, &server);
    printf("Success: started UDP server on port %" PRIu16 "\n", port);
}
----------------------------------------
--------------------------------------------------------------------------------

[2] Score: 0.8763 (cosine: 0.6763, path_boost: 0.2000)
🎯 Path keywords matched: udp, gnrc
📁 File: RIOT/examples/networking/gnrc/gnrc_networking_mac/udp.c
🔖 Chunk: udp_chunk0001
📄 Full Text:
----------------------------------------
/*
 * Copyright (C) 2015-17 Freie Universität Berlin
 *
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v2.1. See the file LICENSE in the top level
 * directory for more details.
 */

/**
 * @ingroup     examples
 * @{
 *
 * @file
 * @brief       Demonstrating the sending and receiving of UDP data
 *
 * @author      Hauke Petersen <hauke.petersen@fu-berlin.de>
 * @author      Martine Lenders <m.lenders@fu-berlin.de>
 *
 * @}
 */

#include <stdio.h>
#include <inttypes.h>

#include "net/gnrc.h"
#include "net/gnrc/ipv6.h"
#include "net/gnrc/netif.h"
#include "net/gnrc/netif/hdr.h"
#include "net/gnrc/udp.h"
#include "net/gnrc/pktdump.h"
#include "shell.h"
#include "timex.h"
#include "utlist.h"
#include "xtimer.h"

static gnrc_netreg_entry_t server = GNRC_NETREG_ENTRY_INIT_PID(GNRC_NETREG_DEMUX_CTX_ALL,
                                                               KERNEL_PID_UNDEF);

static void send(char *addr_str, char *port_str, char *data, unsigned int num,
                 unsigned int delay)
{
    gnrc_netif_t *netif = NULL;
    char *iface;
    uint16_t port;
    ipv6_addr_t addr;

    /* get interface, if available */
    iface = ipv6_addr_split_iface(addr_str);
    if ((!iface) && (gnrc_netif_numof() == 1)) {
        netif = gnrc_netif_iter(NULL);
    }
    else if (iface){
        netif = gnrc_netif_get_by_pid(atoi(iface));
    }
    /* parse destination address */
    if (ipv6_addr_from_str(&addr, addr_str) == NULL) {
        puts("Error: unable to parse destination address");
        return;
    }
    /* parse port */
    port = atoi(port_str);
    if (port == 0) {
        puts("Error: unable to parse destination port");
        return;
    }
----------------------------------------
--------------------------------------------------------------------------------

[3] Score: 0.8420 (cosine: 0.6420, path_boost: 0.2000)
🎯 Path keywords matched: udp, gnrc
📁 File: RIOT/examples/networking/gnrc/gnrc_networking_mac/udp.c
🔖 Chunk: udp_chunk0002
📄 Full Text:
----------------------------------------
for (unsigned int i = 0; i < num; i++) {
        gnrc_pktsnip_t *payload, *udp, *ip;
        unsigned payload_size;
        /* allocate payload */
        payload = gnrc_pktbuf_add(NULL, data, strlen(data), GNRC_NETTYPE_UNDEF);
        if (payload == NULL) {
            puts("Error: unable to copy data to packet buffer");
            return;
        }
        /* store size for output */
        payload_size = (unsigned)payload->size;
        /* allocate UDP header, set source port := destination port */
        udp = gnrc_udp_hdr_build(payload, port, port);
        if (udp == NULL) {
            puts("Error: unable to allocate UDP header");
            gnrc_pktbuf_release(payload);
            return;
        }
        /* allocate IPv6 header */
        ip = gnrc_ipv6_hdr_build(udp, NULL, &addr);
        if (ip == NULL) {
            puts("Error: unable to allocate IPv6 header");
            gnrc_pktbuf_release(udp);
            return;
        }
        /* add netif header, if interface was given */
        if (netif != NULL) {
            gnrc_pktsnip_t *netif_hdr = gnrc_netif_hdr_build(NULL, 0, NULL, 0);
            if (netif_hdr == NULL) {
                puts("Error: unable to allocate netif header");
                gnrc_pktbuf_release(ip);
                return;
            }
            gnrc_netif_hdr_set_netif(netif_hdr->data, netif);
            ip = gnrc_pkt_prepend(ip, netif_hdr);
        }
        /* send packet */
        if (!gnrc_netapi_dispatch_send(GNRC_NETTYPE_UDP, GNRC_NETREG_DEMUX_CTX_ALL, ip)) {
            puts("Error: unable to locate UDP thread");
            gnrc_pktbuf_release(ip);
            return;
        }
        /* access to `payload` was implicitly given up with the send operation above
         * => use temporary variable for output */
        printf("Success: sent %u byte(s) to [%s]:%u\n", payload_size, addr_str,
               port);
        xtimer_usleep(delay);
    }
}
----------------------------------------
--------------------------------------------------------------------------------

[4] Score: 0.8356 (cosine: 0.6356, path_boost: 0.2000)
🎯 Path keywords matched: udp, gnrc
📁 File: RIOT/examples/networking/gnrc/gnrc_networking_mac/udp.c
🔖 Chunk: udp_chunk0004
📄 Full Text:
----------------------------------------
/* check if server is running at all */
    if (server.target.pid == KERNEL_PID_UNDEF) {
        printf("Error: server was not running\n");
        return;
    }
    /* stop server */
    gnrc_netreg_unregister(GNRC_NETTYPE_UDP, &server);
    server.target.pid = KERNEL_PID_UNDEF;
    puts("Success: stopped UDP server");
}
----------------------------------------
--------------------------------------------------------------------------------

[5] Score: 0.7088 (cosine: 0.6088, path_boost: 0.1000)
🎯 Path keywords matched: gnrc
📁 File: RIOT/examples/networking/gnrc/gnrc_networking/README.md
🔖 Chunk: README_chunk0001
📄 Full Text:
----------------------------------------
# gnrc_networking example

This example shows you how to try out the code in two different ways:
Either by communicating between the RIOT machine and its Linux host,
or by communicating between two RIOT instances.
Note that the former only works with native, i.e. if you run RIOT on
your Linux machine.

## Connecting RIOT native and the Linux host

> **Note:** RIOT does not support IPv4, so you need to stick to IPv6
> anytime. To establish a connection between RIOT and the Linux host,
> you will need `netcat` (with IPv6 support). Ubuntu 14.04 comes with
> netcat IPv6 support pre-installed.
> On Debian it's available in the package `netcat-openbsd`. Be aware
> that many programs require you to add an option such as -6 to tell
> them to use IPv6, otherwise they will fail. If you're using a
> _Raspberry Pi_, run `sudo modprobe ipv6` before trying this example,
> because raspbian does not load the IPv6 module automatically.
> On some systems (openSUSE for example), the _firewall_ may interfere,
> and prevent some packets to arrive at the application (they will
> however show up in Wireshark, which can be confusing). So be sure to
> adjust your firewall rules, or turn it off (who needs security
> anyway).

First, make sure you've compiled the application by calling `make`.

Now, create a tap interface:

    sudo ip tuntap add tap0 mode tap user ${USER}
    sudo ip link set tap0 up

Now you can start the `gnrc_networking` example by invoking `make term`.
This should automatically connect to the `tap0` interface. If this
doesn't work for any reason, run make term with the tap0 interface as
the PORT environment variable:

    PORT=tap0 make term

To verify that there is connectivity between RIOT and Linux, go to the
RIOT console and run `ifconfig`:
----------------------------------------
--------------------------------------------------------------------------------

[6] Score: 0.6766 (cosine: 0.5766, path_boost: 0.1000)
🎯 Path keywords matched: udp
📁 File: RIOT/examples/networking/misc/posix_sockets/udp.c
🔖 Chunk: udp_chunk0001
📄 Full Text:
----------------------------------------
/*
 * Copyright (C) 2015 Martine Lenders <mlenders@inf.fu-berlin.de>
 *
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v2.1. See the file LICENSE in the top level
 * directory for more details.
 */

/**
 * @ingroup     examples
 * @{
 *
 * @file
 * @brief       Demonstrating the sending and receiving of UDP data over POSIX sockets.
 *
 * @author      Martine Lenders <mlenders@inf.fu-berlin.de>
 *
 * @}
 */

/* needed for posix usleep */
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#ifdef SOCK_HAS_IPV6
#include "net/ipv6/addr.h"     /* for interface parsing */
#include "net/netif.h"         /* for resolving ipv6 scope */
#endif /* SOCK_HAS_IPV6 */

#include "shell.h"
#include "thread.h"

#define SERVER_MSG_QUEUE_SIZE   (8)
#define SERVER_BUFFER_SIZE      (64)

static int server_socket = -1;
static char server_buffer[SERVER_BUFFER_SIZE];
static char server_stack[THREAD_STACKSIZE_DEFAULT];
static msg_t server_msg_queue[SERVER_MSG_QUEUE_SIZE];
----------------------------------------
--------------------------------------------------------------------------------

[7] Score: 0.6527 (cosine: 0.5527, path_boost: 0.1000)
🎯 Path keywords matched: udp
📁 File: RIOT/examples/networking/misc/posix_sockets/udp.c
🔖 Chunk: udp_chunk0004
📄 Full Text:
----------------------------------------
/* check if server is already running */
    if (server_socket >= 0) {
        puts("Error: server already running");
        return 1;
    }
    /* start server (which means registering pktdump for the chosen port) */
    if (thread_create(server_stack, sizeof(server_stack), THREAD_PRIORITY_MAIN - 1,
                      0,
                      _server_thread, port_str, "UDP server") <= KERNEL_PID_UNDEF) {
        server_socket = -1;
        puts("error initializing thread");
        return 1;
    }
    return 0;
}
----------------------------------------
--------------------------------------------------------------------------------

[8] Score: 0.6350 (cosine: 0.5350, path_boost: 0.1000)
🎯 Path keywords matched: udp
📁 File: RIOT/examples/networking/misc/posix_sockets/udp.c
🔖 Chunk: udp_chunk0005
📄 Full Text:
----------------------------------------
if (argc < 2) {
        printf("usage: %s [send|server]\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "send") == 0) {
        uint32_t num = 1;
        uint32_t delay = 1000000;
        if (argc < 5) {
            printf("usage: %s send <addr> <port> <data> [<num> [<delay in us>]]\n",
                   argv[0]);
            return 1;
        }
        if (argc > 5) {
            num = atoi(argv[5]);
        }
        if (argc > 6) {
            delay = atoi(argv[6]);
        }
        return udp_send(argv[2], argv[3], argv[4], num, delay);
    }
    else if (strcmp(argv[1], "server") == 0) {
        if (argc < 3) {
            printf("usage: %s server [start|stop]\n", argv[0]);
            return 1;
        }
        if (strcmp(argv[2], "start") == 0) {
            if (argc < 4) {
                printf("usage %s server start <port>\n", argv[0]);
                return 1;
            }
            return udp_start_server(argv[3]);
        }
        else {
            puts("error: invalid command");
            return 1;
        }
    }
    else {
        puts("error: invalid command");
        return 1;
    }
}

SHELL_COMMAND(udp, "send data over UDP and listen on UDP ports", _udp_cmd);

/** @} */

**Instructions:**
1. Use the provided RIOT OS documentation chunks and also use the RIOT OS repository chunks to answer the user's question comprehensively, but please focus more on the documentation chunks then on the repository chunks, the chunks from the examples are like additional information that can be used.
2. Focus on practical, actionable information from the documentation
3. Reference specific modules, functions, or APIs mentioned in the chunks when relevant
4. If the chunks don't contain sufficient information, clearly state what information is missing
5. Provide code examples or configuration snippets when available in the chunks
6. Maintain technical accuracy and use RIOT OS terminology correctly
7. Structure your response clearly with headers and bullet points when appropriate

**Response Guidelines:**
- Start with a direct answer to the user's question
- Provide step-by-step instructions when applicable
- Include relevant code examples from the documentation
- Mention any prerequisites or dependencies
- Reference specific file paths or modules when helpful
- End with related concepts or next steps if appropriate

Please provide a comprehensive answer based on the RIOT OS documentation provided above.
:



